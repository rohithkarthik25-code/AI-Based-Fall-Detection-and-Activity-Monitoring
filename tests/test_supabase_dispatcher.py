"""
Unit tests for SupabaseDispatcher.
Validates payload construction, mock network calls, and thread resilience.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.alerts.supabase_dispatcher import SupabaseDispatcher


class TestSupabaseDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = SupabaseDispatcher(
            supabase_url="https://mockproject.supabase.co",
            supabase_key="mock-key-123",
            camera_id="test-cam-01"
        )

    def test_init_parameters(self):
        self.assertEqual(self.dispatcher.camera_id, "test-cam-01")
        self.assertEqual(self.dispatcher.supabase_url, "https://mockproject.supabase.co")
        self.assertEqual(self.dispatcher._headers["apikey"], "mock-key-123")
        self.assertEqual(self.dispatcher._headers["Authorization"], "Bearer mock-key-123")

    @patch("urllib.request.urlopen")
    def test_upload_snapshot_valid(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        url = self.dispatcher.upload_snapshot(frame, person_id=1)

        self.assertIsNotNone(url)
        self.assertTrue(url.startswith("https://mockproject.supabase.co/storage/v1/object/public/fall-snapshots/"))

    def test_upload_snapshot_invalid_frame(self):
        url = self.dispatcher.upload_snapshot(None, person_id=1)
        self.assertIsNone(url)

        empty_frame = np.array([])
        url_empty = self.dispatcher.upload_snapshot(empty_frame, person_id=1)
        self.assertIsNone(url_empty)

    @patch("urllib.request.urlopen")
    def test_post_fall_record_worker(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        # Should execute without throwing any exception
        self.dispatcher._post_fall_record_worker(
            person_id=1,
            severity="SEVERE",
            fall_score=0.92,
            duration_down=3.5,
            frame=None
        )
        self.assertTrue(mock_urlopen.called)

    @patch("threading.Thread")
    def test_dispatch_fall_spawns_thread(self, mock_thread):
        self.dispatcher.dispatch_fall(
            person_id=5,
            severity="MODERATE",
            fall_score=0.78,
            duration_down=2.0
        )
        self.assertTrue(mock_thread.called)


if __name__ == "__main__":
    unittest.main()

