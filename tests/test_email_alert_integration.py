"""
Integration Tests for Email Alert Endpoint with In-Process Fake SMTP and Mock Supabase.
Covers end-to-end HTTP/network flows:
1. Fake in-process SMTP server listening on localhost.
2. Mock Supabase Auth, REST (atomic claim), Storage, and Roles endpoints.
3. Cases:
   - Forged / invalid JWT token -> 401 rejected, 0 emails sent.
   - Another user's event_id -> 403 Forbidden, 0 emails sent, claim reverted.
   - Ownerless event (user_id IS NULL) by standard user -> 403 Forbidden.
   - Ownerless event by Admin -> authorized and email sent to Admin.
   - Legitimate owner calling -> 1 email dispatched and captured by fake SMTP server.
   - Repeated calls -> idempotent skipped status, 0 duplicate emails.
   - Stale CLAIMED row (>2 min) -> successfully reclaimed and dispatched.
"""

import unittest
import threading
import socket
import json
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, parse_qs

from src.alerts.email_alert_service import claim_and_send_email_alert, AuthPermissionError
from src.alerts.alert_dispatcher import AlertDispatcher


# =============================================================================
# 1. In-Process Fake SMTP Server
# =============================================================================

class FakeSMTPServer:
    """Minimal single-threaded in-process SMTP server capturing incoming emails."""
    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, port))
        self.port = self.sock.getsockname()[1]
        self.sock.listen(5)
        self.running = True
        self.received_messages = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.sock.settimeout(0.5)
        while self.running:
            try:
                conn, _ = self.sock.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _handle_client(self, conn):
        try:
            conn.sendall(b"220 localhost FakeSMTP\r\n")
            in_data = False
            data_buffer = []

            while True:
                line = b""
                while not line.endswith(b"\r\n"):
                    chunk = conn.recv(1)
                    if not chunk:
                        return
                    line += chunk

                line_str = line.decode("latin1", errors="replace").strip()

                if in_data:
                    if line_str == ".":
                        in_data = False
                        full_msg = "\r\n".join(data_buffer)
                        self.received_messages.append(full_msg)
                        conn.sendall(b"250 2.0.0 Ok: queued\r\n")
                    else:
                        data_buffer.append(line_str)
                elif line_str.upper().startswith("EHLO") or line_str.upper().startswith("HELO"):
                    conn.sendall(b"250-localhost\r\n250 HELP\r\n")
                elif line_str.upper().startswith("MAIL FROM:"):
                    conn.sendall(b"250 2.1.0 Ok\r\n")
                elif line_str.upper().startswith("RCPT TO:"):
                    conn.sendall(b"250 2.1.5 Ok\r\n")
                elif line_str.upper() == "DATA":
                    in_data = True
                    data_buffer = []
                    conn.sendall(b"354 End data with <CR><LF>.<CR><LF>\r\n")
                elif line_str.upper() == "QUIT":
                    conn.sendall(b"221 2.0.0 Bye\r\n")
                    break
                else:
                    conn.sendall(b"250 Ok\r\n")
        except Exception:
            pass
        finally:
            conn.close()

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


# =============================================================================
# 2. In-Process Mock Supabase HTTP Server
# =============================================================================

class MockSupabaseHandler(BaseHTTPRequestHandler):
    """Mocks Supabase Auth, REST (atomic claim), and Storage API."""

    def log_message(self, format, *args):
        pass  # Suppress HTTP server console logs during tests

    def do_GET(self):
        parsed = urlparse(self.path)

        # 1. Auth Endpoint: /auth/v1/user
        if parsed.path == "/auth/v1/user":
            auth_header = self.headers.get("Authorization", "")
            if "valid-alice-token" in auth_header:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "id": "alice-uuid-1111",
                    "email": "alice@gmail.com"
                }).encode("utf-8"))
            elif "valid-bob-token" in auth_header:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "id": "bob-uuid-2222",
                    "email": "bob@gmail.com"
                }).encode("utf-8"))
            elif "valid-admin-token" in auth_header:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "id": "admin-uuid-9999",
                    "email": "rohithkarthik25@gmail.com"
                }).encode("utf-8"))
            else:
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"message": "Invalid JWT"}')
            return

        # 2. Roles Endpoint: /rest/v1/user_roles
        if parsed.path == "/rest/v1/user_roles":
            qs = parse_qs(parsed.query)
            user_id = qs.get("user_id", [""])[0].replace("eq.", "")
            if user_id == "admin-uuid-9999":
                resp_data = [{"role": "admin"}]
            else:
                resp_data = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
            return

        # 3. Storage Snapshot Download: /storage/v1/object/fall-snapshots/<uuid>.jpg
        if parsed.path.startswith("/storage/v1/object/fall-snapshots/"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(b"\xff\xd8\xff\xe0FAKE_JPEG_IMAGE_CONTENT\xff\xd9")
            return

        self.send_response(404)
        self.end_headers()

    def do_PATCH(self):
        parsed = urlparse(self.path)
        if parsed.path == "/rest/v1/fall_events":
            content_length = int(self.headers.get("Content-Length", 0))
            patch_body = json.loads(self.rfile.read(content_length).decode("utf-8"))
            qs = parse_qs(parsed.query)
            event_id = qs.get("id", [""])[0].replace("eq.", "")

            # Look up event in mock DB state
            event = self.server.db_events.get(event_id)
            if not event:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[]")
                return

            # Check if this is an atomic claim request
            if "email_claimed_at" in patch_body and patch_body.get("email_status") == "CLAIMED":
                now_t = datetime.now(timezone.utc)
                already_sent = event.get("email_sent_at") is not None
                claimed_at = event.get("email_claimed_at")

                is_stale = False
                if claimed_at:
                    c_dt = datetime.fromisoformat(claimed_at)
                    if now_t - c_dt > timedelta(minutes=2):
                        is_stale = True

                # Can claim if not sent AND (not claimed OR claimed > 2 min ago)
                if not already_sent and (claimed_at is None or is_stale):
                    event["email_status"] = "CLAIMED"
                    event["email_claimed_at"] = patch_body["email_claimed_at"]
                    event["email_attempts"] = event.get("email_attempts", 0) + 1
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps([event]).encode("utf-8"))
                    return
                else:
                    # Concurrency block: 0 rows returned
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b"[]")
                    return

            # Update status (e.g. SENT, FAILED)
            event.update(patch_body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps([event]).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


# =============================================================================
# 3. Integration Test Harness
# =============================================================================

class TestEmailAlertIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Start Fake SMTP Server
        cls.smtp_server = FakeSMTPServer()

        # 2. Start Mock Supabase Server
        cls.http_server = HTTPServer(("127.0.0.1", 0), MockSupabaseHandler)
        cls.mock_supabase_url = f"http://127.0.0.1:{cls.http_server.server_port}"
        cls.http_server.db_events = {}
        cls.server_thread = threading.Thread(target=cls.http_server.serve_forever, daemon=True)
        cls.server_thread.start()

        # 3. Configure Dispatcher to point to Fake SMTP
        cls.dispatcher = AlertDispatcher()
        cls.dispatcher.simulation_mode = False
        cls.dispatcher.smtp_host = cls.smtp_server.host
        cls.dispatcher.smtp_port = cls.smtp_server.port
        cls.dispatcher.smtp_user = "test-sender@example.com"
        cls.dispatcher.smtp_pass = "dummy-pass"
        cls.dispatcher.from_name = "FallGuard AI Test"
        cls.dispatcher.use_tls = False  # Local plain fake SMTP

    @classmethod
    def tearDownClass(cls):
        cls.smtp_server.stop()
        cls.http_server.shutdown()

    def setUp(self):
        self.smtp_server.received_messages.clear()
        self.http_server.db_events.clear()

    # -------------------------------------------------------------------------
    # Test Cases
    # -------------------------------------------------------------------------

    def test_case_a_forged_or_invalid_jwt_rejected(self):
        """Case A: Forged or invalid JWT returns 401, 0 emails sent."""
        with self.assertRaises(AuthPermissionError):
            claim_and_send_email_alert(
                event_id="any-event",
                auth_token="Bearer forged-attacker-token-xxx",
                supabase_url=self.mock_supabase_url,
                dispatcher=self.dispatcher
            )
        self.assertEqual(len(self.smtp_server.received_messages), 0)

    def test_case_b_another_users_event_forbidden(self):
        """Case B: User Bob calling for User Alice's event is 403 Forbidden; claim reverted."""
        self.http_server.db_events["alice-event-1"] = {
            "id": "alice-event-1",
            "user_id": "alice-uuid-1111",
            "severity": "SEVERE",
            "fall_score": 0.95,
            "duration_down": 4.5,
            "person_id": 1,
            "snapshot_url": None,
            "email_sent_at": None,
            "email_claimed_at": None
        }

        with self.assertRaises(AuthPermissionError):
            claim_and_send_email_alert(
                event_id="alice-event-1",
                auth_token="Bearer valid-bob-token",
                supabase_url=self.mock_supabase_url,
                dispatcher=self.dispatcher
            )

        self.assertEqual(len(self.smtp_server.received_messages), 0)
        # Verify claim was reverted to FAILED
        self.assertEqual(self.http_server.db_events["alice-event-1"]["email_status"], "FAILED")

    def test_case_c_ownerless_event_by_standard_user_forbidden(self):
        """Case C: Ownerless event (user_id=None) cannot be triggered by standard user."""
        self.http_server.db_events["edge-event-none"] = {
            "id": "edge-event-none",
            "user_id": None,
            "severity": "SEVERE",
            "fall_score": 0.88,
            "email_sent_at": None,
            "email_claimed_at": None
        }

        with self.assertRaises(AuthPermissionError):
            claim_and_send_email_alert(
                event_id="edge-event-none",
                auth_token="Bearer valid-alice-token",
                supabase_url=self.mock_supabase_url,
                dispatcher=self.dispatcher
            )
        self.assertEqual(len(self.smtp_server.received_messages), 0)

    def test_case_d_ownerless_event_by_admin_authorized(self):
        """Case D: Ownerless event (Python edge cam) triggered by Admin succeeds."""
        self.http_server.db_events["edge-event-admin"] = {
            "id": "edge-event-admin",
            "user_id": None,
            "severity": "SEVERE",
            "fall_score": 0.92,
            "person_id": 1,
            "duration_down": 3.0,
            "snapshot_url": None,
            "email_sent_at": None,
            "email_claimed_at": None
        }

        result = claim_and_send_email_alert(
            event_id="edge-event-admin",
            auth_token="Bearer valid-admin-token",
            supabase_url=self.mock_supabase_url,
            service_role_key="mock-service-key",
            dispatcher=self.dispatcher
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["recipient"], "rohithkarthik25@gmail.com")
        self.assertEqual(len(self.smtp_server.received_messages), 1)

    def test_case_e_legitimate_owner_sends_email_with_attachment(self):
        """Case E: Legitimate owner triggers email, received by SMTP with snapshot."""
        valid_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        snapshot_url = f"https://qbzuwqanxnkpeuioiiao.supabase.co/storage/v1/object/public/fall-snapshots/{valid_uuid}.jpg"

        self.http_server.db_events["alice-real-event"] = {
            "id": "alice-real-event",
            "user_id": "alice-uuid-1111",
            "severity": "SEVERE",
            "fall_score": 0.99,
            "person_id": 1,
            "duration_down": 6.2,
            "snapshot_url": snapshot_url,
            "email_sent_at": None,
            "email_claimed_at": None
        }

        result = claim_and_send_email_alert(
            event_id="alice-real-event",
            auth_token="Bearer valid-alice-token",
            supabase_url=self.mock_supabase_url,
            dispatcher=self.dispatcher
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["recipient"], "alice@gmail.com")

        # Verify email arrived in fake SMTP server
        self.assertEqual(len(self.smtp_server.received_messages), 1)
        email_content = self.smtp_server.received_messages[0]
        self.assertIn("To: alice@gmail.com", email_content)
        self.assertIn("EMERGENCY ALERT", email_content)
        self.assertIn("Person #1", email_content)

        # Verify database record updated to SENT
        self.assertEqual(self.http_server.db_events["alice-real-event"]["email_status"], "SENT")
        self.assertIsNotNone(self.http_server.db_events["alice-real-event"]["email_sent_at"])

    def test_case_f_repeated_call_idempotent_no_duplicate(self):
        """Case F: Second call for already sent event returns skipped, 0 extra emails."""
        self.http_server.db_events["already-sent-event"] = {
            "id": "already-sent-event",
            "user_id": "alice-uuid-1111",
            "severity": "MILD",
            "fall_score": 0.60,
            "email_sent_at": datetime.now(timezone.utc).isoformat(),
            "email_claimed_at": datetime.now(timezone.utc).isoformat(),
            "email_status": "SENT"
        }

        result = claim_and_send_email_alert(
            event_id="already-sent-event",
            auth_token="Bearer valid-alice-token",
            supabase_url=self.mock_supabase_url,
            dispatcher=self.dispatcher
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(len(self.smtp_server.received_messages), 0)

    def test_case_g_stale_claimed_row_reclaimed_and_sent(self):
        """Case G: Event stuck in CLAIMED from 5 minutes ago is reclaimed and dispatched."""
        three_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        self.http_server.db_events["stale-event"] = {
            "id": "stale-event",
            "user_id": "alice-uuid-1111",
            "severity": "MODERATE",
            "fall_score": 0.75,
            "person_id": 2,
            "duration_down": 2.0,
            "snapshot_url": None,
            "email_sent_at": None,
            "email_claimed_at": three_min_ago,
            "email_status": "CLAIMED",
            "email_attempts": 1
        }

        result = claim_and_send_email_alert(
            event_id="stale-event",
            auth_token="Bearer valid-alice-token",
            supabase_url=self.mock_supabase_url,
            dispatcher=self.dispatcher
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(len(self.smtp_server.received_messages), 1)
        self.assertEqual(self.http_server.db_events["stale-event"]["email_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
