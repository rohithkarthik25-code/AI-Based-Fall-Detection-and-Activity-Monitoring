"""
Instant Email Alert Dispatcher (SMTP).
Dispatches asynchronous email notifications with annotated fall snapshots,
severity levels, biomechanical scores, and recovery status.
Executes non-blocking network calls via a background thread pool.
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "simulation_mode": True,  # When True, logs alerts locally without requiring live API keys
    "cooldown_seconds": 30.0,
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
        "sender_email": "",
        "sender_password": "",
        "recipient_email": "",
        "from_name": "FallGuard AI",
    },
}


def _load_dotenv(filepath: str = ".env") -> None:
    """Lightweight zero-dependency .env loader."""
    candidates = [
        filepath,
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]
    for cp in candidates:
        if os.path.exists(cp):
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass


class AlertDispatcher:
    """
    Non-blocking email alert dispatcher for fall incidents.
    Sends SMTP Email alerts (HTML + embedded snapshot).
    """

    def __init__(
        self,
        config_path: Optional[str] = "alert_config.json",
        enabled: bool = True,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.config = self._load_config(config_path)

        # Check if caller explicitly passed a custom config file (e.g. in tests)
        custom_sim = None
        if config_path and os.path.exists(config_path) and config_path != "alert_config.json":
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    custom_sim = json.load(f).get("simulation_mode")
            except Exception:
                pass

        _load_dotenv()

        # Environment variable overrides
        # Support both SMTP_EMAIL/SMTP_PASSWORD and SMTP_USER/SMTP_APP_PASSWORD
        smtp_user = os.getenv("SMTP_USER") or os.getenv("SMTP_EMAIL")
        smtp_pass = os.getenv("SMTP_APP_PASSWORD") or os.getenv("SMTP_PASSWORD")
        smtp_host = os.getenv("SMTP_HOST") or "smtp.gmail.com"
        smtp_port = int(os.getenv("SMTP_PORT") or 587)
        alert_recipient = os.getenv("ALERT_RECIPIENT_EMAIL") or smtp_user
        from_name = os.getenv("ALERT_FROM_NAME") or "FallGuard AI"

        if smtp_user and smtp_pass and alert_recipient:
            self.config["email"]["enabled"] = True
            self.config["email"]["smtp_server"] = smtp_host
            self.config["email"]["smtp_port"] = smtp_port
            self.config["email"]["sender_email"] = smtp_user
            self.config["email"]["sender_password"] = smtp_pass
            self.config["email"]["recipient_email"] = alert_recipient
            self.config["email"]["from_name"] = from_name
            self.config["simulation_mode"] = False

        if custom_sim is not None:
            self.config["simulation_mode"] = custom_sim

        self.last_alert_time: Dict[int, float] = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="AlertDispatcher")
        self.snapshot_dir = "fall_logs"
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Loads configuration from JSON file or initializes defaults."""
        cfg = DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    cfg.update(loaded)
            except Exception as e:
                print(f"[AlertDispatcher] Warning: Could not read config file {config_path}: {e}")
        return cfg

    def dispatch_fall_alert(
        self,
        frame: Optional[np.ndarray],
        person_id: int,
        severity: str,
        score: float,
        duration: float = 0.0,
        activity: str = "Unknown",
        location: str = "Camera 1 (Main Feed)",
    ) -> bool:
        """
        Asynchronously dispatches a confirmed fall alert to configured channels.
        Returns True if the alert was scheduled, False if throttled by cooldown.
        """
        if not self.enabled or not self.config.get("enabled", True):
            return False

        now = time.time()
        last_t = self.last_alert_time.get(person_id, 0.0)
        cooldown = self.config.get("cooldown_seconds", self.cooldown_seconds)

        if (now - last_t) < cooldown:
            return False  # Cooldown active, suppress duplicate spam

        self.last_alert_time[person_id] = now
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Fast memory copy of frame so main thread is not blocked by disk I/O
        frame_copy = frame.copy() if (frame is not None and frame.size > 0) else None

        # Dispatch via thread pool non-blockingly
        self.executor.submit(
            self._send_payload_worker,
            frame_raw=frame_copy,
            person_id=person_id,
            severity=severity,
            score=score,
            duration=duration,
            activity=activity,
            location=location,
            timestamp_str=timestamp_str,
        )
        return True

    def dispatch_recovery_alert(self, person_id: int, duration: float) -> None:
        """Dispatches notification when a fallen person has successfully stood up."""
        if not self.enabled or not self.config.get("enabled", True):
            return

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.executor.submit(
            self._send_recovery_worker,
            person_id=person_id,
            duration=duration,
            timestamp_str=timestamp_str,
        )

    def _annotate_snapshot(
        self,
        image: np.ndarray,
        person_id: int,
        severity: str,
        score: float,
        timestamp_str: str,
    ) -> np.ndarray:
        """Draws prominent emergency banner at top of snapshot before sending."""
        h, w = image.shape[:2]
        banner_h = 42
        cv2.rectangle(image, (0, 0), (w, banner_h), (20, 20, 80), -1)
        cv2.line(image, (0, banner_h), (w, banner_h), (40, 50, 240), 2)

        header_text = f"EMERGENCY: FALL CONFIRMED - P{person_id} | {severity} | Score: {score:.2f}"
        cv2.putText(image, header_text, (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, timestamp_str, (max(12, w - 190), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)
        return image

    def _send_payload_worker(
        self,
        frame_raw: Optional[np.ndarray],
        person_id: int,
        severity: str,
        score: float,
        duration: float,
        activity: str,
        location: str,
        timestamp_str: str,
    ) -> None:
        """Background thread worker for disk persistence and network dispatch."""
        sim_mode = self.config.get("simulation_mode", False)
        email_cfg = self.config.get("email", {})

        # Asynchronously annotate and write snapshot on background thread
        snapshot_path = None
        if frame_raw is not None and frame_raw.size > 0:
            annotated_snapshot = self._annotate_snapshot(frame_raw, person_id, severity, score, timestamp_str)
            filename = f"alert_fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}_p{person_id}.jpg"
            snapshot_path = os.path.join(self.snapshot_dir, filename)
            cv2.imwrite(snapshot_path, annotated_snapshot)

        if sim_mode or not email_cfg.get("enabled"):
            print("\n" + "=" * 60)
            print("[ALERT DISPATCHER - SIMULATION MODE]")
            print(f"Alert successfully generated for Person {person_id}!")
            print(f"Details : {severity} fall (score={score:.2f}) at {timestamp_str}")
            if snapshot_path:
                print(f"Snapshot: {snapshot_path}")
            print("Configure alert_config.json to send live Email alerts.")
            print("=" * 60 + "\n")
            return

        # ── Dispatch to Email / SMTP ──
        if email_cfg.get("enabled") and email_cfg.get("sender_email") and email_cfg.get("recipient_email"):
            self._send_email(
                cfg=email_cfg,
                person_id=person_id,
                severity=severity,
                score=score,
                duration=duration,
                activity=activity,
                location=location,
                timestamp_str=timestamp_str,
                snapshot_path=snapshot_path,
            )

    def _send_email(
        self,
        cfg: Dict[str, Any],
        person_id: int,
        severity: str,
        score: float,
        duration: float,
        activity: str,
        location: str,
        timestamp_str: str,
        snapshot_path: Optional[str],
    ) -> None:
        """Sends rich HTML alert email with embedded snapshot attachment via SMTP."""
        sender = cfg.get("sender_email")
        password = cfg.get("sender_password")
        recipient = cfg.get("recipient_email")
        server_host = cfg.get("smtp_server", "smtp.gmail.com")
        port = int(cfg.get("smtp_port", 587))
        use_tls = cfg.get("use_tls", True)

        try:
            from_name = cfg.get("from_name", "FallGuard AI")
            msg = MIMEMultipart("related")
            msg["Subject"] = f"🚨 [CRITICAL ALERT] Fall Detected: Person #{person_id} ({severity})"
            msg["From"] = f"{from_name} <{sender}>" if from_name else sender
            msg["To"] = recipient

            # HTML Body
            html = f"""
            <html>
              <body style="font-family: Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px;">
                <div style="max-width: 600px; margin: auto; background-color: #ffffff; border-radius: 8px; border: 1px solid #e0e4e8; overflow: hidden;">
                  <div style="background-color: #d32f2f; color: #ffffff; padding: 16px 20px;">
                    <h2 style="margin: 0; font-size: 20px;">CRITICAL EMERGENCY: FALL DETECTED</h2>
                    <p style="margin: 4px 0 0; font-size: 13px; opacity: 0.9;">AI Safety Monitoring System Alert</p>
                  </div>
                  <div style="padding: 20px;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                      <tr><td style="padding: 8px 0; color: #666;">Subject:</td><td style="font-weight: bold;">Person #{person_id}</td></tr>
                      <tr><td style="padding: 8px 0; color: #666;">Severity:</td><td style="font-weight: bold; color: #d32f2f;">{severity}</td></tr>
                      <tr><td style="padding: 8px 0; color: #666;">Fall Confidence Score:</td><td style="font-weight: bold;">{score:.2f} / 1.00</td></tr>
                      <tr><td style="padding: 8px 0; color: #666;">Pre-Fall Activity:</td><td>{activity.upper()}</td></tr>
                      <tr><td style="padding: 8px 0; color: #666;">Location:</td><td>{location}</td></tr>
                      <tr><td style="padding: 8px 0; color: #666;">Timestamp:</td><td>{timestamp_str}</td></tr>
                    </table>
                    <div style="margin-top: 18px; text-align: center;">
                      <p style="color: #444; font-size: 13px; margin-bottom: 8px;"><b>Incident Snapshot:</b></p>
                      <img src="cid:snapshot_img" style="max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px;" alt="Fall Incident Snapshot"/>
                    </div>
                  </div>
                  <div style="background-color: #f9fafb; padding: 12px 20px; font-size: 12px; color: #888; border-top: 1px solid #eef0f2;">
                    This is an automated safety alert generated by the AI Fall Detection & Activity Monitoring System.
                  </div>
                </div>
              </body>
            </html>
            """
            msg_alt = MIMEMultipart("alternative")
            msg.attach(msg_alt)
            msg_alt.attach(MIMEText(html, "html"))

            # Attach image
            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as img_f:
                    mime_img = MIMEImage(img_f.read(), name=os.path.basename(snapshot_path))
                    mime_img.add_header("Content-ID", "<snapshot_img>")
                    mime_img.add_header("Content-Disposition", "inline", filename=os.path.basename(snapshot_path))
                    msg.attach(mime_img)

            # SMTP Connect & Send
            if use_tls:
                server = smtplib.SMTP(server_host, port, timeout=12)
                server.starttls(context=ssl.create_default_context())
            else:
                server = smtplib.SMTP_SSL(server_host, port, timeout=12)

            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
            server.quit()
            print(f"[AlertDispatcher] Email alert dispatched successfully to {recipient}.")
        except Exception as e:
            print(f"[AlertDispatcher] Email dispatch failed: {e}")

    def send_email_alert(
        self,
        recipient_email: str,
        event_id: str,
        person_id: int,
        severity: str,
        fall_score: float,
        duration_down: float,
        event_time: str,
        snapshot_bytes: Optional[bytes] = None
    ) -> bool:
        """Sends an emergency alert email directly to the specified recipient."""
        if getattr(self, "simulation_mode", False):
            print(f"[AlertDispatcher - SIMULATION] Email alert to {recipient_email} for Person #{person_id}")
            return True

        host = getattr(self, "smtp_host", None) or os.environ.get("SMTP_HOST", "smtp.gmail.com")
        port = int(getattr(self, "smtp_port", None) or os.environ.get("SMTP_PORT", 587))
        user = getattr(self, "smtp_user", None) or os.environ.get("SMTP_USER", "")
        password = getattr(self, "smtp_pass", None) or os.environ.get("SMTP_APP_PASSWORD", "")
        from_name = getattr(self, "from_name", None) or os.environ.get("ALERT_FROM_NAME", "FallGuard AI")
        use_tls = getattr(self, "use_tls", True) if getattr(self, "use_tls", None) is not None else (port == 587)

        msg = MIMEMultipart("related")
        msg["Subject"] = f"🚨 [EMERGENCY ALERT] Fall Detected: Person #{person_id} ({severity})"
        msg["From"] = f"{from_name} <{user}>" if user else from_name
        msg["To"] = recipient_email

        score_str = f"{fall_score * 100:.0f}%" if fall_score <= 1.0 else f"{fall_score:.1f}%"
        html = f"""
        <html><body>
        <h2>EMERGENCY ALERT: Fall Detected</h2>
        <p>Person #{person_id} | Severity: {severity} | Score: {score_str}</p>
        <p>Dispatched to: {recipient_email}</p>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        if snapshot_bytes:
            img = MIMEImage(snapshot_bytes, _subtype="jpeg", name="snapshot.jpg")
            img.add_header("Content-ID", "<snapshot_img>")
            msg.attach(img)

        s = smtplib.SMTP(host, port, timeout=10)
        try:
            if use_tls:
                s.starttls(context=ssl.create_default_context())
            if user and password:
                try:
                    s.login(user, password)
                except Exception:
                    pass
            s.sendmail(user or "alert@localhost", recipient_email, msg.as_string())
            return True
        finally:
            try:
                s.quit()
            except Exception:
                pass

    def _send_recovery_worker(self, person_id: int, duration: float, timestamp_str: str) -> None:
        """Worker for recording recovery events."""
        print(
            f"[AlertDispatcher] Recovery recorded for Person #{person_id} at {timestamp_str} "
            f"(floor time {duration:.1f}s)."
        )

    def close(self) -> None:
        """Gracefully shuts down the background thread pool."""
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
