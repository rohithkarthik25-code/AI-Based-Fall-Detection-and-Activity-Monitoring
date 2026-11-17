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

    @patch("urllib.request.urlopen")
    def test_upload_snapshot_uses_uuid_filename(self, mock_urlopen):
        """Filename must satisfy the storage policy / insert trigger '<uuid>.jpg' regex."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        url1 = self.dispatcher.upload_snapshot(frame, person_id=1)
        url2 = self.dispatcher.upload_snapshot(frame, person_id=1)

        pattern = (
            r"^https://mockproject\.supabase\.co/storage/v1/object/public/fall-snapshots/"
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg$"
        )
        self.assertRegex(url1, pattern)
        self.assertRegex(url2, pattern)
        self.assertNotEqual(url1, url2)

        upload_req = mock_urlopen.call_args_list[0][0][0]
        self.assertEqual(upload_req.get_header("X-upsert"), "false")

    @patch("src.alerts.supabase_dispatcher._load_dotenv")
    def test_key_resolution_prefers_service_role_env(self, _mock_dotenv):
        with patch.dict("os.environ", {"SUPABASE_SERVICE_ROLE_KEY": "svc-role-test"}):
            d = SupabaseDispatcher(supabase_url="https://mockproject.supabase.co")
        self.assertEqual(d.supabase_key, "svc-role-test")
        self.assertEqual(d._headers["Authorization"], "Bearer svc-role-test")

    @patch("src.alerts.supabase_dispatcher._load_dotenv")
    def test_key_resolution_explicit_key_wins(self, mock_dotenv):
        with patch.dict("os.environ", {"SUPABASE_SERVICE_ROLE_KEY": "svc-role-test"}):
            d = SupabaseDispatcher(supabase_url="https://mockproject.supabase.co", supabase_key="explicit")
        self.assertEqual(d.supabase_key, "explicit")
        mock_dotenv.assert_not_called()

    @patch("src.alerts.supabase_dispatcher._load_dotenv")
    def test_key_resolution_falls_back_to_anon(self, _mock_dotenv):
        from src.alerts import supabase_dispatcher as mod
        with patch.dict("os.environ", {}, clear=True):
            d = SupabaseDispatcher(supabase_url="https://mockproject.supabase.co")
        self.assertEqual(d.supabase_key, mod.DEFAULT_SUPABASE_KEY)

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

