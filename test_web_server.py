"""
Unit tests verifying FastAPI endpoints, telemetry API, and HTML delivery.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

import web_server
from web_pipeline import WebVideoPipeline


class TestWebServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize pipeline in dummy/synthetic mode
        cls.pipeline = WebVideoPipeline(
            source="__dummy__",
            enable_alerts=False,
            enable_3d=False,
            enable_activity=False,
        )
        web_server.pipeline = cls.pipeline
        cls.client = TestClient(web_server.app)

    @classmethod
    def tearDownClass(cls):
        if cls.pipeline:
            cls.pipeline.stop()

    def test_index_page_returns_html(self):
        """Verify that GET / returns the modern monitoring dashboard."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AI FALL DETECTION & BIOMECHANICAL MONITOR", response.text)
        self.assertIn("/video_feed", response.text)

    def test_telemetry_endpoint(self):
        """Verify that GET /api/telemetry returns valid JSON metrics including 3D landmarks."""
        response = self.client.get("/api/telemetry")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active_persons", data)
        self.assertIn("fall_active", data)
        self.assertIn("fps", data)
        self.assertIn("focus_world_landmarks", data)
        self.assertIn("persons", data)

    def test_websocket_telemetry(self):
        """Verify that /ws/telemetry streams real-time JSON updates for Three.js."""
        with self.client.websocket_connect("/ws/telemetry") as ws:
            data = ws.receive_json()
            self.assertIn("active_persons", data)
            self.assertIn("fall_active", data)
            self.assertIn("focus_world_landmarks", data)

    def test_focus_endpoint(self):
        """Verify that POST /api/focus successfully sets focus person ID."""
        response = self.client.post("/api/focus", json={"person_id": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["focus_person_id"], 2)

        # Test reset to auto
        response_auto = self.client.post("/api/focus", json={"person_id": None})
        self.assertEqual(response_auto.status_code, 200)
        self.assertIsNone(response_auto.json()["focus_person_id"])

    def test_incidents_endpoint(self):
        """Verify that GET /api/incidents returns a list."""
        response = self.client.get("/api/incidents")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)


if __name__ == "__main__":
    unittest.main()
