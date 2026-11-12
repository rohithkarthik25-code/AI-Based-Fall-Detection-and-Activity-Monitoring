"""
Integration test suite for Step 3.
Validates FallDetector end-to-end orchestration, 3D side-panel rendering,
activity monitoring integration, and backwards-compatibility flags.
"""

import os
import unittest
import numpy as np
import cv2

from fall_detector import FallDetector
from dashboard import Dashboard, draw_person_box


class TestStep3Integration(unittest.TestCase):

    def test_detector_initialization(self):
        """Test default and customized initialization of FallDetector."""
        # Full mode (3D + Activity)
        det_full = FallDetector(enable_3d=True, enable_activity=True, display=False)
        self.assertTrue(det_full.enable_3d)
        self.assertTrue(det_full.enable_activity)
        self.assertIsNotNone(det_full.visualizer_3d)
        self.assertIsNotNone(det_full.activity_monitor)

        # Legacy 2D mode (backwards compatible)
        det_legacy = FallDetector(enable_3d=False, enable_activity=False, display=False)
        self.assertFalse(det_legacy.enable_3d)
        self.assertFalse(det_legacy.enable_activity)
        self.assertIsNone(det_legacy.visualizer_3d)
        self.assertIsNone(det_legacy.activity_monitor)

    def test_dashboard_with_activity(self):
        """Test Dashboard drawing with and without activity metadata."""
        dash = Dashboard()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Without activity summary
        out1 = dash.draw(frame.copy(), active_persons=1, fps=28.5)
        self.assertEqual(out1.shape, (480, 640, 3))

        # With activity summary
        summary = {
            "activity_label": "Walking",
            "activity_confidence": 0.88,
            "time_percentages": {"Sitting": 20.0, "Standing": 50.0, "Walking": 30.0},
        }
        out2 = dash.draw(frame.copy(), active_persons=1, fps=28.5, summary_state=summary)
        self.assertEqual(out2.shape, (480, 640, 3))

    def test_draw_person_box_with_activity(self):
        """Test bounding box drawing with activity badge."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "severity": "NONE",
            "fall_score": 0.05,
            "fall_detected": False,
            "activity_label": "Sitting",
            "torso_angle": 12.0,
            "signal_scores": {},
        }
        box = [100, 100, 200, 300]
        # Should execute cleanly without error
        draw_person_box(frame, box, pid=1, state=state)
        self.assertGreater(np.count_nonzero(frame), 100)

    def test_synthetic_video_processing(self):
        """Test processing a tiny synthetic video file end-to-end."""
        test_video = "__test_synth.mp4"
        test_out = "__test_synth_out.mp4"

        try:
            # Create a 10-frame synthetic video (640x480)
            writer = cv2.VideoWriter(test_video, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 480))
            for i in range(10):
                f = np.full((480, 640, 3), i * 20, dtype=np.uint8)
                writer.write(f)
            writer.release()

            detector = FallDetector(
                output_path=test_out,
                display=False,
                enable_3d=True,
                enable_activity=True,
            )

            res = detector.process(test_video)
            self.assertTrue(os.path.exists(test_out))
            self.assertGreater(os.path.getsize(test_out), 1000)

        finally:
            if os.path.exists(test_video):
                os.remove(test_video)
            if os.path.exists(test_out):
                os.remove(test_out)


if __name__ == "__main__":
    unittest.main()

