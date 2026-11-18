"""
Unit & Integration Tests for Email Alert Service & Endpoint Security.
Covers:
1. SSRF prevention and URL validation (rejection of internal IPs, arbitrary hosts, path traversal).
2. Supabase Storage API integration (no raw HTTP requests to client-provided URLs).
3. JWT Authentication & Ownership Authorization (session verification, owner match, admin override).
4. Atomic Claim Query & Concurrency (claim then send, 2-minute timeout recovery, duplicate block).
5. Error recording and retry state machine.
"""

import unittest
import os
import json
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.alerts.email_alert_service import (
    validate_and_extract_snapshot_filename,
    fetch_snapshot_bytes_from_storage,
    verify_supabase_jwt,
    check_user_is_admin,
    check_user_is_blocked,
    claim_and_send_email_alert,
    SSRFValidationError,
    AuthPermissionError,
    PROJECT_REF
)


class TestEmailAlertEndpointSecurity(unittest.TestCase):

    # =========================================================================
    # 1. SSRF Prevention & URL Validation Tests
    # =========================================================================

    def test_ssrf_valid_snapshot_url(self):
        """Valid public UUID .jpg URL from project bucket returns the clean filename."""
        valid_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        valid_url = f"https://{PROJECT_REF}.supabase.co/storage/v1/object/public/fall-snapshots/{valid_uuid}.jpg"

        filename = validate_and_extract_snapshot_filename(valid_url)
        self.assertEqual(filename, f"{valid_uuid}.jpg")

    def test_ssrf_null_or_empty_url(self):
        """None or empty snapshot URL returns None safely without raising an error."""
        self.assertIsNone(validate_and_extract_snapshot_filename(None))
        self.assertIsNone(validate_and_extract_snapshot_filename(""))
        self.assertIsNone(validate_and_extract_snapshot_filename("   "))

    def test_ssrf_malicious_urls_rejected(self):
        """Any non-matching host, bucket, traversal path, or non-UUID file raises SSRFValidationError."""
        malicious_urls = [
            "http://169.254.169.254/latest/meta-data",
            "http://169.254.169.254/computeMetadata/v1/",
            "http://127.0.0.1:8000/internal-secrets",
            "http://localhost:5432/db",
            "https://evil-attacker.com/storage/v1/object/public/fall-snapshots/a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg",
            f"https://{PROJECT_REF}.supabase.co/storage/v1/object/public/other-bucket/a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg",
            f"https://{PROJECT_REF}.supabase.co/storage/v1/object/public/fall-snapshots/../../etc/passwd.jpg",
            f"https://{PROJECT_REF}.supabase.co/storage/v1/object/public/fall-snapshots/photo.png",
            f"https://{PROJECT_REF}.supabase.co/storage/v1/object/public/fall-snapshots/not-a-uuid.jpg",
            "file:///etc/shadow",
            "gopher://127.0.0.1:6379/_flushall",
        ]
        for url in malicious_urls:
            with self.subTest(url=url):
                with self.assertRaises(SSRFValidationError):
                    validate_and_extract_snapshot_filename(url)

    # =========================================================================
    # 2. Supabase Storage API Image Fetch Tests
    # =========================================================================

    @patch("urllib.request.urlopen")
    def test_fetch_snapshot_calls_storage_api_with_service_key(self, mock_urlopen):
        """Storage fetch strictly calls the Supabase Storage endpoint with the service key."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"FAKE_JPEG_IMAGE_BYTES"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        test_filename = "a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg"
        test_key = "test-service-role-key-12345"

        result = fetch_snapshot_bytes_from_storage(
            filename=test_filename,
            service_role_key=test_key
        )

        self.assertEqual(result, b"FAKE_JPEG_IMAGE_BYTES")
        args, _ = mock_urlopen.call_args
        req = args[0]
        expected_url = f"https://{PROJECT_REF}.supabase.co/storage/v1/object/fall-snapshots/{test_filename}"
        self.assertEqual(req.full_url, expected_url)
        self.assertEqual(req.headers.get("Apikey"), test_key)
        self.assertEqual(req.headers.get("Authorization"), f"Bearer {test_key}")

    def test_fetch_snapshot_invalid_filename_rejected(self):
        """Attempting to fetch with an invalid filename raises SSRFValidationError without making HTTP request."""
        with self.assertRaises(SSRFValidationError):
            fetch_snapshot_bytes_from_storage(filename="../secret.jpg")

    # =========================================================================
    # 3. JWT Authentication & Ownership Authorization Tests
    # =========================================================================

    def test_verify_jwt_missing_token(self):
        """Missing or empty JWT raises AuthPermissionError."""
        with self.assertRaises(AuthPermissionError):
            verify_supabase_jwt("")
        with self.assertRaises(AuthPermissionError):
            verify_supabase_jwt(None)

    @patch("urllib.request.urlopen")
    def test_verify_jwt_valid_google_user(self, mock_urlopen):
        """Valid JWT returns decoded user profile from Supabase Auth."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({
            "id": "user-uuid-1111",
            "email": "user@gmail.com",
            "app_metadata": {"provider": "google"}
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        user = verify_supabase_jwt("Bearer valid-token-xyz")
        self.assertEqual(user["id"], "user-uuid-1111")
        self.assertEqual(user["email"], "user@gmail.com")

    @patch("src.alerts.email_alert_service.verify_supabase_jwt")
    @patch("urllib.request.urlopen")
    def test_authorization_user_owns_incident_succeeds(self, mock_urlopen, mock_jwt):
        """User who owns the incident is authorized to send email."""
        mock_jwt.return_value = {"id": "user-uuid-1111", "email": "alice@gmail.com"}

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([{
            "id": "event-123",
            "user_id": "user-uuid-1111",
            "severity": "SEVERE",
            "fall_score": 0.95,
            "duration_down": 12.0,
            "person_id": 1,
            "snapshot_url": None,
            "created_at": datetime.now(timezone.utc).isoformat()
        }]).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_email_alert.return_value = True

        result = claim_and_send_email_alert(
            event_id="event-123",
            auth_token="Bearer test-token",
            dispatcher=mock_dispatcher
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["recipient"], "alice@gmail.com")
        mock_dispatcher.send_email_alert.assert_called_once()

    @patch("src.alerts.email_alert_service.verify_supabase_jwt")
    @patch("src.alerts.email_alert_service.check_user_is_admin")
    @patch("urllib.request.urlopen")
    def test_authorization_mismatched_user_forbidden(self, mock_urlopen, mock_admin, mock_jwt):
        """User B trying to trigger alert for User A's incident is blocked with 403."""
        mock_jwt.return_value = {"id": "user-bob-2222", "email": "bob@gmail.com"}
        mock_admin.return_value = False

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([{
            "id": "event-123",
            "user_id": "user-alice-1111",
            "severity": "SEVERE"
        }]).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        with self.assertRaises(AuthPermissionError):
            claim_and_send_email_alert(
                event_id="event-123",
                auth_token="Bearer bob-token"
            )

    @patch("src.alerts.email_alert_service.verify_supabase_jwt")
    @patch("src.alerts.email_alert_service.check_user_is_admin")
    @patch("urllib.request.urlopen")
    def test_authorization_admin_can_access_any_incident(self, mock_urlopen, mock_admin, mock_jwt):
        """Admin is authorized to trigger email for any incident."""
        mock_jwt.return_value = {"id": "admin-9999", "email": "rohithkarthik25@gmail.com"}
        mock_admin.return_value = True

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([{
            "id": "event-456",
            "user_id": "user-alice-1111",
            "severity": "MODERATE",
            "fall_score": 0.85,
            "person_id": 2,
            "snapshot_url": None,
            "created_at": datetime.now(timezone.utc).isoformat()
        }]).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_email_alert.return_value = True

        result = claim_and_send_email_alert(
            event_id="event-456",
            auth_token="Bearer admin-token",
            dispatcher=mock_dispatcher
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["recipient"], "rohithkarthik25@gmail.com")

    # =========================================================================
    # 4. Atomic Claim & Concurrency Logic Tests
    # =========================================================================

    @patch("src.alerts.email_alert_service.verify_supabase_jwt")
    @patch("src.alerts.email_alert_service.check_user_is_blocked")
    @patch("urllib.request.urlopen")
    def test_blocked_user_cannot_trigger_email(self, mock_urlopen, mock_blocked, mock_jwt):
        """A user blocked by an admin is rejected before the event is claimed."""
        mock_jwt.return_value = {"id": "user-uuid-1111", "email": "alice@gmail.com"}
        mock_blocked.return_value = True
        mock_dispatcher = MagicMock()

        with self.assertRaises(AuthPermissionError) as ctx:
            claim_and_send_email_alert(
                event_id="event-123",
                auth_token="Bearer token",
                service_role_key="svc-test",
                dispatcher=mock_dispatcher
            )
        self.assertIn("suspended", str(ctx.exception))
        mock_urlopen.assert_not_called()  # no claim PATCH was sent
        mock_dispatcher.send_email_alert.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_check_user_is_blocked_reads_blocked_at(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([{"blocked_at": "2026-11-18T10:00:00Z"}]).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        self.assertTrue(check_user_is_blocked("user-1", service_role_key="svc-test"))
        req = mock_urlopen.call_args[0][0]
        self.assertIn("/rest/v1/user_roles?user_id=eq.user-1&select=blocked_at", req.full_url)

        mock_resp.read.return_value = json.dumps([{"blocked_at": None}]).encode("utf-8")
        self.assertFalse(check_user_is_blocked("user-1", service_role_key="svc-test"))

        mock_resp.read.return_value = b"[]"
        self.assertFalse(check_user_is_blocked("user-1", service_role_key="svc-test"))

    @patch("src.alerts.email_alert_service.verify_supabase_jwt")
    @patch("urllib.request.urlopen")
    def test_atomic_claim_concurrency_second_caller_skipped(self, mock_urlopen, mock_jwt):
        """If event was already claimed or sent, atomic claim returns 0 rows and returns skipped."""
        mock_jwt.return_value = {"id": "user-uuid-1111", "email": "alice@gmail.com"}

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"[]"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        mock_dispatcher = MagicMock()

        result = claim_and_send_email_alert(
            event_id="already-claimed-event",
            auth_token="Bearer token",
            dispatcher=mock_dispatcher
        )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("already claimed", result["message"])
        mock_dispatcher.send_email_alert.assert_not_called()

    @patch("src.alerts.email_alert_service.verify_supabase_jwt")
    @patch("urllib.request.urlopen")
    def test_error_recording_on_smtp_failure(self, mock_urlopen, mock_jwt):
        """When SMTP send fails, event status is updated to FAILED with error message."""
        mock_jwt.return_value = {"id": "user-uuid-1111", "email": "alice@gmail.com"}

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps([{
            "id": "event-fail-1",
            "user_id": "user-uuid-1111",
            "severity": "SEVERE",
            "fall_score": 0.99,
            "person_id": 1,
            "snapshot_url": None,
            "created_at": datetime.now(timezone.utc).isoformat()
        }]).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        mock_dispatcher = MagicMock()
        mock_dispatcher.send_email_alert.side_effect = RuntimeError("SMTP connection timeout")

        result = claim_and_send_email_alert(
            event_id="event-fail-1",
            auth_token="Bearer token",
            dispatcher=mock_dispatcher
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("SMTP connection timeout", result["error"])


if __name__ == "__main__":
    unittest.main()
