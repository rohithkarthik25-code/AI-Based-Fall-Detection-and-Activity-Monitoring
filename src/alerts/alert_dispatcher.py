"""
Instant Multi-Channel Alert Dispatcher (Telegram Bot & Email/SMTP).
Dispatches asynchronous notifications with annotated fall snapshots,
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
import requests


DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "simulation_mode": True,  # When True, logs alerts locally without requiring live API keys
    "cooldown_seconds": 30.0,
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
    },
    "email": {
        "enabled": False,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "use_tls": True,
        "sender_email": "",
        "sender_password": "",
        "recipient_email": "",
    },
}


class AlertDispatcher:
    """
    Non-blocking, multi-channel alert dispatcher for fall incidents.
    Supports Telegram Bot (text + snapshot) and SMTP Email (HTML + embedded snapshot).
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

        # Environment variable overrides
        if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"):
            self.config["telegram"]["enabled"] = True
            self.config["telegram"]["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")
            self.config["telegram"]["chat_id"] = os.getenv("TELEGRAM_CHAT_ID")
            self.config["simulation_mode"] = False

        if os.getenv("SMTP_EMAIL") and os.getenv("SMTP_PASSWORD") and os.getenv("ALERT_RECIPIENT_EMAIL"):
            self.config["email"]["enabled"] = True
            self.config["email"]["sender_email"] = os.getenv("SMTP_EMAIL")
            self.config["email"]["sender_password"] = os.getenv("SMTP_PASSWORD")
            self.config["email"]["recipient_email"] = os.getenv("ALERT_RECIPIENT_EMAIL")
            self.config["simulation_mode"] = False

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
        tg_cfg = self.config.get("telegram", {})
        email_cfg = self.config.get("email", {})

        # Asynchronously annotate and write snapshot on background thread
        snapshot_path = None
        if frame_raw is not None and frame_raw.size > 0:
            annotated_snapshot = self._annotate_snapshot(frame_raw, person_id, severity, score, timestamp_str)
            filename = f"alert_fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}_p{person_id}.jpg"
            snapshot_path = os.path.join(self.snapshot_dir, filename)
            cv2.imwrite(snapshot_path, annotated_snapshot)

        # Message formulation
        msg_text = (
            f"🚨 *CRITICAL EMERGENCY ALERT: FALL DETECTED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Subject ID* : Person #{person_id}\n"
            f"⚠️ *Severity*   : {severity}\n"
            f"📊 *Fall Score* : {score:.2f} / 1.00\n"
            f"⏱️ *Duration*   : {duration:.1f}s on ground\n"
            f"🏃 *Last State* : {activity.upper()}\n"
            f"📍 *Location*   : {location}\n"
            f"🕒 *Timestamp*  : {timestamp_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Action Required: Check subject immediately."
        )

        if sim_mode or (not tg_cfg.get("enabled") and not email_cfg.get("enabled")):
            print("\n" + "=" * 60)
            print("[ALERT DISPATCHER - SIMULATION MODE]")
            print(f"Alert successfully generated for Person {person_id}!")
            print(f"Details : {severity} fall (score={score:.2f}) at {timestamp_str}")
            if snapshot_path:
                print(f"Snapshot: {snapshot_path}")
            print("Configure alert_config.json to send to live Telegram/Email.")
            print("=" * 60 + "\n")
            return

        # ── 1. Dispatch to Telegram Bot ──
        if tg_cfg.get("enabled") and tg_cfg.get("bot_token") and tg_cfg.get("chat_id"):
            self._send_telegram(
                bot_token=tg_cfg["bot_token"],
                chat_id=tg_cfg["chat_id"],
                caption=msg_text,
                snapshot_path=snapshot_path,
            )

        # ── 2. Dispatch to Email / SMTP ──
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

    def _send_telegram(
        self,
        bot_token: str,
        chat_id: str,
        caption: str,
        snapshot_path: Optional[str],
    ) -> None:
        """Sends alert message with photo attachment via Telegram Bot API."""
        try:
            if snapshot_path and os.path.exists(snapshot_path):
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                with open(snapshot_path, "rb") as photo_file:
                    files = {"photo": photo_file}
                    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}
                    resp = requests.post(url, files=files, data=data, timeout=10)
            else:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                data = {"chat_id": chat_id, "text": caption, "parse_mode": "Markdown"}
                resp = requests.post(url, data=data, timeout=10)

            if resp.status_code == 200:
                print(f"[AlertDispatcher] Telegram alert dispatched successfully to chat {chat_id}.")
            else:
                print(f"[AlertDispatcher] Telegram dispatch error ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"[AlertDispatcher] Telegram dispatch failed: {e}")

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
            msg = MIMEMultipart("related")
            msg["Subject"] = f"🚨 [CRITICAL ALERT] Fall Detected: Person #{person_id} ({severity})"
            msg["From"] = sender
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

    def _send_recovery_worker(self, person_id: int, duration: float, timestamp_str: str) -> None:
        """Worker for sending recovery notifications."""
        sim_mode = self.config.get("simulation_mode", False)
        tg_cfg = self.config.get("telegram", {})

        msg = (
            f"✅ *STATUS UPDATE: SUBJECT RECOVERED*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Subject ID* : Person #{person_id}\n"
            f"⏱️ *Floor Time* : {duration:.1f} seconds\n"
            f"🕒 *Timestamp*  : {timestamp_str}\n"
            f"Status: Subject has successfully returned to an upright/normal posture."
        )

        if sim_mode or not tg_cfg.get("enabled"):
            print(f"[AlertDispatcher - SIMULATION] Recovery recorded for Person #{person_id} at {timestamp_str}.")
            return

        if tg_cfg.get("enabled") and tg_cfg.get("bot_token") and tg_cfg.get("chat_id"):
            try:
                url = f"https://api.telegram.org/bot{tg_cfg['bot_token']}/sendMessage"
                data = {"chat_id": tg_cfg["chat_id"], "text": msg, "parse_mode": "Markdown"}
                requests.post(url, data=data, timeout=8)
            except Exception as e:
                print(f"[AlertDispatcher] Recovery notification failed: {e}")

    def close(self) -> None:
        """Gracefully shuts down the background thread pool."""
        try:
            self.executor.shutdown(wait=False)
        except Exception:
            pass
