"""
Supabase Real-Time Alert & Telemetry Dispatcher for AI Fall Detection.
Integrates Edge Camera telemetry, fall incident records, and snapshot image uploads
with Supabase PostgreSQL, Storage, and Edge Functions.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
import urllib.request
import urllib.error
import cv2
import numpy as np

DEFAULT_SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qbzuwqanxnkpeuioiiao.supabase.co")
DEFAULT_SUPABASE_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFienV3cWFueG5rcGV1aW9paWFvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTQ2MjYxMzMsImV4cCI6MjExMDIwMjEzM30.cC9AZlgWMe3doYp2DtxjYRUGdDVn5L7YmTHpiKjRP0Q"
)


class SupabaseDispatcher:
    def __init__(
        self,
        supabase_url: str = DEFAULT_SUPABASE_URL,
        supabase_key: str = DEFAULT_SUPABASE_KEY,
        camera_id: str = "camera-edge-01"
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.camera_id = camera_id
        self._headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        self.logged_events: set[tuple[int, str]] = set()

    def upload_snapshot(self, frame: np.ndarray, person_id: int) -> str | None:
        """Encodes frame as JPEG and uploads to Supabase Storage 'fall-snapshots' bucket."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return None

        try:
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                return None

            jpeg_bytes = buffer.tobytes()
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"incident_{self.camera_id}_p{person_id}_{timestamp_str}.jpg"

            upload_url = f"{self.supabase_url}/storage/v1/object/fall-snapshots/{filename}"
            upload_headers = {
                "apikey": self.supabase_key,
                "Authorization": f"Bearer {self.supabase_key}",
                "Content-Type": "image/jpeg",
                "x-upsert": "true"
            }

            req = urllib.request.Request(upload_url, data=jpeg_bytes, headers=upload_headers, method="POST")
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status in (200, 201):
                    public_url = f"{self.supabase_url}/storage/v1/object/public/fall-snapshots/{filename}"
                    return public_url
        except Exception as e:
            print(f"[SupabaseDispatcher] Snapshot upload warning: {e}")
        return None

    def _post_fall_record_worker(
        self,
        person_id: int,
        severity: str,
        fall_score: float,
        duration_down: float,
        frame: np.ndarray | None
    ) -> None:
        try:
            snapshot_url = None
            if frame is not None:
                snapshot_url = self.upload_snapshot(frame, person_id)

            payload = {
                "person_id": int(person_id),
                "severity": severity,
                "fall_score": round(float(fall_score), 2),
                "duration_down": round(float(duration_down), 2),
                "snapshot_url": snapshot_url,
                "resolved": False,
                "notes": f"Detected by {self.camera_id}"
            }

            rest_url = f"{self.supabase_url}/rest/v1/fall_events"
            req = urllib.request.Request(
                rest_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status in (200, 201):
                    print(f"[SupabaseDispatcher] Successfully synced fall event to Supabase cloud! Person #{person_id} ({severity})")
                else:
                    print(f"[SupabaseDispatcher] Supabase response status: {resp.status}")
        except Exception as e:
            print(f"[SupabaseDispatcher] Error dispatching to Supabase: {e}")

    def dispatch_fall(
        self,
        person_id: int,
        severity: str,
        fall_score: float,
        duration_down: float = 0.0,
        frame: np.ndarray | None = None
    ) -> None:
        """Asynchronously dispatches a confirmed fall event to Supabase in a background thread."""
        minute_key = (person_id, datetime.now(timezone.utc).strftime("%Y%m%d%H%M"))
        if minute_key in self.logged_events:
            return
        self.logged_events.add(minute_key)

        frame_copy = frame.copy() if frame is not None else None
        thread = threading.Thread(
            target=self._post_fall_record_worker,
            args=(person_id, severity, fall_score, duration_down, frame_copy),
            daemon=True
        )
        thread.start()

    def send_heartbeat(self, fps: float, active_persons: int = 1, status: str = "ONLINE") -> None:
        """Sends periodic camera telemetry heartbeat."""
        def _worker():
            try:
                payload = {
                    "camera_id": self.camera_id,
                    "fps": round(float(fps), 2),
                    "active_persons": int(active_persons),
                    "status": status,
                    "last_heartbeat": datetime.now(timezone.utc).isoformat()
                }
                url = f"{self.supabase_url}/rest/v1/camera_telemetry"
                headers = dict(self._headers)
                headers["Prefer"] = "resolution=merge-duplicates"

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=5) as _:
                    pass
            except Exception as e:
                # Silent failure for non-critical telemetry heartbeat
                pass

        threading.Thread(target=_worker, daemon=True).start()

