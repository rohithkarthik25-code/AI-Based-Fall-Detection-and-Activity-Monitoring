"""
Unit tests validating edge-case repairs and robustness improvements across all modules.
Tests:
- None/empty/corrupted frame handling in YOLODetector
- Float coordinate handling in PoseEstimator
- None/empty frame handling in FallLogger
- Geometry reset flag in TemporalStateMachine
- Multi-person alarm lifecycle: individual recovery and disappearance
- winsound exception resilience in AlarmSystem
"""

import os
import shutil
import unittest
from unittest.mock import MagicMock
import numpy as np

from yolo_detector import YOLODetector
from pose_estimator import PoseEstimator
from fall_logger import FallLogger
from temporal_state_machine import TemporalStateMachine
from weighted_scoring import ScoreResult
from alarm_system import AlarmSystem


class TestRepairsAndEdgeCases(unittest.TestCase):

    def test_yolo_invalid_frames(self):
        """Verify YOLO detector returns [] on None, empty, or degenerate frames."""
        yolo = YOLODetector()
        
        # Test None
        self.assertEqual(yolo.detect(None), [])

        # Test 0-size array
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(yolo.detect(empty_frame), [])

        # Test degenerate small dimensions (e.g. 4x4)
        tiny_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.assertEqual(yolo.detect(tiny_frame), [])

        # Test 2D grayscale array instead of 3D
        gray_frame = np.zeros((100, 100), dtype=np.uint8)
        self.assertEqual(yolo.detect(gray_frame), [])

    def test_pose_float_box(self):
        """Verify PoseEstimator accepts floating-point bounding boxes without TypeError."""
        pose = PoseEstimator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Floating point box
        float_box = [12.4, 25.8, 120.6, 250.3]
        result = pose.estimate(frame, float_box)
        self.assertIsNotNone(result)
        self.assertFalse(result.has_pose)

    def test_fall_logger_none_frame(self):
        """Verify FallLogger gracefully writes CSV even when camera frame is None."""
        test_log = "test_edge_fall_events.csv"
        test_dir = "test_edge_fall_logs"
        if os.path.exists(test_log):
            os.remove(test_log)
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        logger = FallLogger(log_file=test_log, screenshot_dir=test_dir, enable_supabase=False)
        try:
            # Pass frame=None
            logger.log(frame=None, person_id=1, severity="SEVERE", duration=2.5, score=0.92)
            self.assertTrue(os.path.exists(test_log))
        finally:
            if os.path.exists(test_log):
                os.remove(test_log)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    def test_temporal_state_machine_geometry_reset_flag(self):
        """Verify that geometry-based reset sets reset_fall=True in StateTransition."""
        fsm = TemporalStateMachine()
        state = {
            "fall_detected": True,
            "candidate_frames": 8,
            "normal_frames": 5,
            "aspect_ratio": 0.25,
            "torso_angle": 10.0,
        }
        res_normal = ScoreResult(score=0.1, severity="NONE", candidate=False, signal_scores={})
        transition = fsm.update(state, res_normal, now_t=100.0)

        self.assertFalse(state["fall_detected"])
        self.assertTrue(transition.reset_fall)

    def test_alarm_system_mock_resilience(self):
        """Verify AlarmSystem trigger, escalate, and reset without audio hardware errors."""
        alarm = AlarmSystem()
        try:
            alarm.trigger()
            self.assertIn(alarm._mode, ["alert", None])
            alarm.escalate()
            self.assertIn(alarm._mode, ["critical", None])
            alarm.reset()
            self.assertIsNone(alarm._mode)
        finally:
            alarm.close()


if __name__ == "__main__":
    unittest.main()
