"""
Email Alert Service for Supabase Auth & Google OAuth Fall Detection Notifications.
Handles SSRF-safe snapshot resolution, atomic database event claiming,
JWT authentication/authorization, and SMTP dispatch to verified Google email.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone, timedelta
import urllib.request
import urllib.error
import json
from typing import Any

from src.alerts.alert_dispatcher import AlertDispatcher

PROJECT_REF = "qbzuwqanxnkpeuioiiao"
DEFAULT_SUPABASE_URL = f"https://{PROJECT_REF}.supabase.co"
SNAPSHOT_URL_PATTERN = re.compile(
    rf"^https://{re.escape(PROJECT_REF)}\.supabase\.co/storage/v1/object/public/fall-snapshots/([0-9a-fA-F]{{8}}-[0-9a-fA-F]{{4}}-[0-9a-fA-F]{{4}}-[0-9a-fA-F]{{4}}-[0-9a-fA-F]{{12}}\.jpg)$"
)


class SSRFValidationError(ValueError):
    """Raised when a snapshot URL fails strict domain/bucket/UUID validation."""
    pass


class AuthPermissionError(PermissionError):
    """Raised when an authenticated user does not have permission to access the incident."""
    pass


def validate_and_extract_snapshot_filename(snapshot_url: str | None) -> str | None:
    """
    Validates that a snapshot URL strictly matches the allowed Supabase bucket public URL
    with a UUID filename. Never allows arbitrary URLs, internal IP addresses, or path traversals.

    Returns the clean UUID filename (e.g. '11111111-2222-3333-4444-555555555555.jpg') or None.
    Raises SSRFValidationError on any mismatch.
    """
    if not snapshot_url or not snapshot_url.strip():
        return None

    clean_url = snapshot_url.strip()
    match = SNAPSHOT_URL_PATTERN.match(clean_url)
    if not match:
        raise SSRFValidationError(
            f"Invalid snapshot URL: URL must match https://{PROJECT_REF}.supabase.co/storage/v1/object/public/fall-snapshots/<uuid>.jpg"
        )
    return match.group(1)


def fetch_snapshot_bytes_from_storage(
    filename: str,
    supabase_url: str = DEFAULT_SUPABASE_URL,
    service_role_key: str | None = None
) -> bytes | None:
    """
    Downloads the snapshot image strictly from the Supabase Storage API using the service role key.
    Never fetches from client-provided URLs or makes external HTTP requests.
    """
    if not filename:
        return None

    # Sanity check filename is strict UUID + .jpg
    if not re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\.jpg$", filename):
        raise SSRFValidationError(f"Invalid storage filename: {filename}")

    key = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    target_url = f"{supabase_url.rstrip('/')}/storage/v1/object/fall-snapshots/{filename}"

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}"
    } if key else {}

    req = urllib.request.Request(target_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                return resp.read()
    except Exception as e:
        print(f"[EmailAlertService] Storage fetch error for {filename}: {e}")
        return None
    return None


def verify_supabase_jwt(token: str, supabase_url: str = DEFAULT_SUPABASE_URL) -> dict[str, Any]:
    """
    Validates a caller's JWT by directly calling Supabase Auth (auth.getUser(token)).
    Never trusts client-decoded JWT claims alone.
    """
    if not token or not token.strip():
        raise AuthPermissionError("Missing Authorization token.")

    clean_token = token.replace("Bearer ", "").strip()
    url = f"{supabase_url.rstrip('/')}/auth/v1/user"
    anon_key = os.environ.get(
        "SUPABASE_ANON_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFienV3cWFueG5rcGV1aW9paWFvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTQ2MjYxMzMsImV4cCI6MjExMDIwMjEzM30.cC9AZlgWMe3doYp2DtxjYRUGdDVn5L7YmTHpiKjRP0Q"
    )

    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {clean_token}"
    }

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data
    except urllib.error.HTTPError as e:
        raise AuthPermissionError(f"Invalid or expired JWT: {e.code}") from e
    except Exception as e:
        raise AuthPermissionError(f"Auth verification error: {e}") from e

    raise AuthPermissionError("Failed to verify user session.")


def check_user_is_admin(user_id: str, supabase_url: str = DEFAULT_SUPABASE_URL, service_role_key: str | None = None) -> bool:
    """
    Queries public.user_roles using service role to check if a user is an admin.
    """
    key = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not key:
        return False

    url = f"{supabase_url.rstrip('/')}/rest/v1/user_roles?user_id=eq.{user_id}&role=eq.admin&select=role"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return len(data) > 0
    except Exception as e:
        print(f"[EmailAlertService] Admin check error for {user_id}: {e}")
    return False


def claim_and_send_email_alert(
    event_id: str,
    auth_token: str,
    supabase_url: str = DEFAULT_SUPABASE_URL,
    service_role_key: str | None = None,
    dispatcher: AlertDispatcher | None = None
) -> dict[str, Any]:
    """
    Orchestrates the entire idempotent email alert flow:
    1. Authenticates JWT with Supabase Auth.
    2. Atomically claims the event using 'claim then send' query.
    3. Validates event ownership (caller matches user_id or is admin).
    4. Validates snapshot URL against SSRF rules.
    5. Dispatches email to caller's verified Google email.
    6. Updates database status to SENT or FAILED.
    """
    key = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    # Step 1: Verify JWT
    user_data = verify_supabase_jwt(auth_token, supabase_url=supabase_url)
    caller_id = user_data.get("id")
    caller_email = user_data.get("email")

    if not caller_id or not caller_email:
        raise AuthPermissionError("Could not determine verified user identity.")

    # Step 2: Atomic Claim
    # UPDATE public.fall_events SET email_claimed_at = now(), email_status = 'CLAIMED', email_attempts = email_attempts + 1
    # WHERE id = event_id AND email_sent_at IS NULL AND (email_claimed_at IS NULL OR email_claimed_at < now() - interval '2 minutes')
    two_minutes_ago = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    claim_url = (
        f"{supabase_url.rstrip('/')}/rest/v1/fall_events?"
        f"id=eq.{event_id}&"
        f"email_sent_at=is.null&"
        f"or=(email_claimed_at.is.null,email_claimed_at.lt.{two_minutes_ago})"
    )

    # Note: In PostgREST REST API, we execute patch
    patch_payload = {
        "email_status": "CLAIMED",
        "email_claimed_at": datetime.now(timezone.utc).isoformat()
    }

    req = urllib.request.Request(
        claim_url,
        data=json.dumps(patch_payload).encode("utf-8"),
        headers=headers,
        method="PATCH"
    )

    claimed_events = []
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status in (200, 201):
                claimed_events = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[EmailAlertService] Atomic claim request error: {e}")
        return {"status": "error", "message": f"Claim failed: {e}"}

    if not claimed_events:
        return {
            "status": "skipped",
            "message": "Event is already claimed, already sent, or within 2-minute lock timeout."
        }

    event = claimed_events[0]
    event_owner_id = event.get("user_id")

    # Step 3: Check authorization
    is_admin = check_user_is_admin(caller_id, supabase_url=supabase_url, service_role_key=key)
    if event_owner_id != caller_id and not is_admin:
        # Revert claim if permission denied
        _revert_claim_payload = {"email_status": "FAILED", "email_error": "Forbidden: user does not own event"}
        revert_req = urllib.request.Request(
            f"{supabase_url.rstrip('/')}/rest/v1/fall_events?id=eq.{event_id}",
            data=json.dumps(_revert_claim_payload).encode("utf-8"),
            headers=headers,
            method="PATCH"
        )
        try:
            with urllib.request.urlopen(revert_req, timeout=5) as _:
                pass
        except Exception:
            pass
        raise AuthPermissionError("Forbidden: You do not have permission to trigger alerts for this incident.")

    # Step 4: SSRF-Safe Snapshot Handling
    snapshot_url = event.get("snapshot_url")
    snapshot_bytes = None
    if snapshot_url:
        try:
            filename = validate_and_extract_snapshot_filename(snapshot_url)
            if filename:
                snapshot_bytes = fetch_snapshot_bytes_from_storage(
                    filename, supabase_url=supabase_url, service_role_key=key
                )
        except SSRFValidationError as e:
            print(f"[EmailAlertService] SSRF Warning: Malicious or invalid snapshot_url rejected: {e}")
            snapshot_bytes = None

    # Step 5: Send Email
    active_dispatcher = dispatcher or AlertDispatcher()
    event_time = event.get("created_at") or datetime.now(timezone.utc).isoformat()
    person_id = event.get("person_id", 1)
    severity = event.get("severity", "SEVERE")
    fall_score = float(event.get("fall_score", 0.95))
    duration_down = float(event.get("duration_down", 0.0))

    try:
        success = active_dispatcher.send_email_alert(
            recipient_email=caller_email,
            event_id=event_id,
            person_id=person_id,
            severity=severity,
            fall_score=fall_score,
            duration_down=duration_down,
            event_time=event_time,
            snapshot_bytes=snapshot_bytes
        )

        if success:
            # Mark SENT
            sent_payload = {
                "email_status": "SENT",
                "email_sent_at": datetime.now(timezone.utc).isoformat(),
                "email_recipient": caller_email,
                "email_error": None
            }
            update_req = urllib.request.Request(
                f"{supabase_url.rstrip('/')}/rest/v1/fall_events?id=eq.{event_id}",
                data=json.dumps(sent_payload).encode("utf-8"),
                headers=headers,
                method="PATCH"
            )
            with urllib.request.urlopen(update_req, timeout=8) as _:
                pass
            return {"status": "sent", "recipient": caller_email, "event_id": event_id}
        else:
            raise RuntimeError("Email dispatcher returned False.")
    except Exception as e:
        # Mark FAILED for retry
        err_msg = str(e)
        failed_payload = {
            "email_status": "FAILED",
            "email_error": err_msg
        }
        fail_req = urllib.request.Request(
            f"{supabase_url.rstrip('/')}/rest/v1/fall_events?id=eq.{event_id}",
            data=json.dumps(failed_payload).encode("utf-8"),
            headers=headers,
            method="PATCH"
        )
        try:
            with urllib.request.urlopen(fail_req, timeout=8) as _:
                pass
        except Exception:
            pass
        return {"status": "failed", "error": err_msg, "event_id": event_id}
