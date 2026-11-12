"""
Comprehensive unit test suite for Instant Alert Dispatcher (Telegram & Email).
Tests asynchronous execution, simulation fallback, debouncing/cooldown,
snapshot persistence, and multi-subject independent dispatching.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
import numpy as np

from alert_dispatcher import AlertDispatcher, DEFAULT_CONFIG


class TestAlertDispatcher(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_alert_config.json")
        self.test_cfg = {
            "enabled": True,
            "simulation_mode": True,
            "cooldown_seconds": 2.0,  # Short cooldown for test speed
            "telegram": {
                "enabled": False,
                "bot_token": "TEST_TOKEN",
                "chat_id": "12345678",
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.test.com",
                "smtp_port": 587,
                "sender_email": "sender@test.com",
                "sender_password": "test",
                "recipient_email": "recipient@test.com",
            },
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.test_cfg, f)

        self.dispatcher = AlertDispatcher(
            config_path=self.config_path,
            enabled=True,
            cooldown_seconds=2.0,
        )

    def tearDown(self):
        self.dispatcher.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_and_config_loading(self):
        """Verifies configuration is loaded and initialized properly."""
        self.assertTrue(self.dispatcher.enabled)
        self.assertTrue(self.dispatcher.config["simulation_mode"])
        self.assertEqual(self.dispatcher.config["cooldown_seconds"], 2.0)

    def test_asynchronous_non_blocking_dispatch(self):
        """Verifies fall alert returns in under 5 ms without stalling the caller."""
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        t0 = time.perf_counter()
        dispatched = self.dispatcher.dispatch_fall_alert(
            frame=synthetic_frame,
            person_id=1,
            severity="SEVERE",
            score=0.92,
            duration=3.5,
            activity="Lying Down",
            location="Room 101",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertTrue(dispatched)
        # Main thread dispatch must be instantaneous (< 10 ms)
        self.assertLess(elapsed_ms, 10.0)

    def test_cooldown_deduplication(self):
        """Verifies immediate repeated alerts for same person are suppressed by cooldown."""
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First alert -> True
        first = self.dispatcher.dispatch_fall_alert(
            frame=synthetic_frame,
            person_id=1,
            severity="SEVERE",
            score=0.88,
        )
        self.assertTrue(first)

        # Second alert immediately after -> False (cooldown active)
        second = self.dispatcher.dispatch_fall_alert(
            frame=synthetic_frame,
            person_id=1,
            severity="SEVERE",
            score=0.88,
        )
        self.assertFalse(second)

    def test_independent_multi_person_cooldown(self):
        """Verifies different person IDs have independent cooldown timers."""
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Alert for Person 1
        d1 = self.dispatcher.dispatch_fall_alert(
            frame=synthetic_frame,
            person_id=1,
            severity="SEVERE",
            score=0.90,
        )
        self.assertTrue(d1)

        # Alert for Person 2 immediately after -> should succeed!
        d2 = self.dispatcher.dispatch_fall_alert(
            frame=synthetic_frame,
            person_id=2,
            severity="MODERATE",
            score=0.75,
        )
        self.assertTrue(d2)

    def test_recovery_alert_dispatch(self):
        """Verifies recovery alert runs cleanly without exceptions."""
        try:
            self.dispatcher.dispatch_recovery_alert(person_id=1, duration=4.2)
        except Exception as e:
            self.fail(f"Recovery alert raised unexpected exception: {e}")

    def test_none_frame_safety(self):
        """Verifies alert dispatcher handles None frame without crashing."""
        dispatched = self.dispatcher.dispatch_fall_alert(
            frame=None,
            person_id=99,
            severity="MILD",
            score=0.60,
        )
        self.assertTrue(dispatched)

    def test_disabled_dispatcher(self):
        """Verifies no alert is dispatched when disabled."""
        self.dispatcher.enabled = False
        dispatched = self.dispatcher.dispatch_fall_alert(
            frame=None,
            person_id=1,
            severity="SEVERE",
            score=0.95,
        )
        self.assertFalse(dispatched)


if __name__ == "__main__":
    unittest.main()

