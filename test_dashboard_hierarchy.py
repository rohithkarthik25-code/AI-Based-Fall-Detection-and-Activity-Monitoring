"""
Unit test suite verifying all 10 dashboard states, information hierarchy,
emergency alerts, and visual boundary styling.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
import numpy as np
import cv2

from dashboard import Dashboard, draw_person_box, _draw_factor_panel, _draw_skeleton
from visualizer_3d import SkeletonVisualizer3D
from pipeline_profiler import PipelineProfiler


class TestDashboardHierarchy(unittest.TestCase):

    def setUp(self):
        self.dash = Dashboard()
        self.vis_3d = SkeletonVisualizer3D(width=320, height=480)
        self.profiler = PipelineProfiler(rolling_window=10, enabled=True)
        # Seed profiler with sample frame times
        for _ in range(6):
            self.profiler.start_frame()
            time.sleep(0.001)
            self.profiler.end_frame()

    # ─────────────────────────────────────────────────────────────
    # State 1: Normal Standing State
    # ─────────────────────────────────────────────────────────────
    def test_state_1_standing(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "activity_label": "Standing",
            "activity_confidence": 0.98,
            "fall_detected": False,
            "candidate_frames": 0,
            "fall_score": 0.05,
            "severity": "NONE",
            "person_id": 1,
            "time_percentages": {"Sitting": 10.0, "Standing": 80.0, "Walking": 10.0},
        }
        out = self.dash.draw(frame, active_persons=1, fps=30.0, summary_state=state, profiler=self.profiler)
        self.assertEqual(out.shape, (480, 640, 3))
        # Ensure frame modified (HUD drawn)
        self.assertTrue(np.any(out > 0))

    # ─────────────────────────────────────────────────────────────
    # State 2: Sitting State
    # ─────────────────────────────────────────────────────────────
    def test_state_2_sitting(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "activity_label": "Sitting",
            "activity_confidence": 0.95,
            "fall_detected": False,
            "candidate_frames": 0,
            "fall_score": 0.12,
            "severity": "NONE",
            "person_id": 1,
            "time_percentages": {"Sitting": 70.0, "Standing": 20.0, "Walking": 10.0},
        }
        out = self.dash.draw(frame, active_persons=1, fps=28.5, summary_state=state, profiler=self.profiler)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue(np.any(out > 0))

    # ─────────────────────────────────────────────────────────────
    # State 3: Walking State
    # ─────────────────────────────────────────────────────────────
    def test_state_3_walking(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "activity_label": "Walking",
            "activity_confidence": 0.92,
            "fall_detected": False,
            "candidate_frames": 0,
            "fall_score": 0.08,
            "severity": "NONE",
            "person_id": 1,
        }
        out = self.dash.draw(frame, active_persons=1, fps=30.2, summary_state=state)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue(np.any(out > 0))

    # ─────────────────────────────────────────────────────────────
    # State 4: Bending State (Warning)
    # ─────────────────────────────────────────────────────────────
    def test_state_4_bending_warning(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "activity_label": "Bending",
            "activity_confidence": 0.88,
            "fall_detected": False,
            "candidate_frames": 3,
            "fall_score": 0.48,
            "severity": "MILD",
            "person_id": 1,
        }
        out = self.dash.draw(frame, active_persons=1, fps=27.0, summary_state=state, profiler=self.profiler)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue(np.any(out > 0))

    # ─────────────────────────────────────────────────────────────
    # State 5: Lying-down State
    # ─────────────────────────────────────────────────────────────
    def test_state_5_lying_down(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "activity_label": "Lying Down",
            "activity_confidence": 0.91,
            "fall_detected": False,
            "candidate_frames": 0,
            "fall_score": 0.20,
            "severity": "NONE",
            "person_id": 1,
        }
        out = self.dash.draw(frame, active_persons=1, fps=29.0, summary_state=state)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue(np.any(out > 0))

    # ─────────────────────────────────────────────────────────────
    # State 6: Confirmed Fall (Emergency Status)
    # ─────────────────────────────────────────────────────────────
    def test_state_6_confirmed_fall(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "activity_label": "Fallen",
            "activity_confidence": 0.99,
            "fall_detected": True,
            "severity": "SEVERE",
            "fall_duration": 4.5,
            "fall_score": 0.89,
            "person_id": 1,
        }
        self.dash.update_fall("SEVERE")
        out = self.dash.draw(frame, active_persons=1, fps=25.0, summary_state=state, profiler=self.profiler)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue(self.dash.fall_active)
        self.assertEqual(self.dash.total_falls, 1)

    # ─────────────────────────────────────────────────────────────
    # State 7: Recovery
    # ─────────────────────────────────────────────────────────────
    def test_state_7_recovery(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = {
            "activity_label": "Standing",
            "activity_confidence": 0.90,
            "fall_detected": False,
            "fall_reset_time": time.time() - 2.0,  # 2 seconds ago
            "fall_score": 0.15,
            "severity": "NONE",
            "person_id": 1,
        }
        self.dash.fall_active = False
        out = self.dash.draw(frame, active_persons=1, fps=29.5, summary_state=state)
        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue(np.any(out > 0))

    # ─────────────────────────────────────────────────────────────
    # State 8: Multiple-Person Detection
    # ─────────────────────────────────────────────────────────────
    def test_state_8_multiple_persons(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state1 = {
            "activity_label": "Standing",
            "activity_confidence": 0.96,
            "fall_detected": False,
            "candidate_frames": 0,
            "fall_score": 0.05,
            "severity": "NONE",
            "person_id": 1,
        }
        state2 = {
            "activity_label": "Walking",
            "activity_confidence": 0.91,
            "fall_detected": False,
            "candidate_frames": 0,
            "fall_score": 0.09,
            "severity": "NONE",
            "person_id": 2,
        }
        tracked = [
            (1, [100, 100, 200, 350], state1),
            (2, [350, 120, 460, 380], state2),
        ]
        draw_person_box(frame, tracked[0][1], pid=1, state=state1)
        draw_person_box(frame, tracked[1][1], pid=2, state=state2)
        out = self.dash.draw(frame, active_persons=2, fps=26.0, summary_state=state1, tracked_persons=tracked)
        self.assertEqual(out.shape, (480, 640, 3))

    # ─────────────────────────────────────────────────────────────
    # State 9: Missing Activity Information
    # ─────────────────────────────────────────────────────────────
    def test_state_9_missing_activity(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # summary_state is completely None or empty
        out_none = self.dash.draw(frame.copy(), active_persons=0, fps=30.0, summary_state=None)
        self.assertEqual(out_none.shape, (480, 640, 3))

        out_empty = self.dash.draw(frame.copy(), active_persons=1, fps=30.0, summary_state={})
        self.assertEqual(out_empty.shape, (480, 640, 3))

    # ─────────────────────────────────────────────────────────────
    # State 10: Missing Fall Information
    # ─────────────────────────────────────────────────────────────
    def test_state_10_missing_fall_info(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Sparse state missing severity, score, duration, etc.
        sparse_state = {"activity_label": "Unknown"}
        out = self.dash.draw(frame, active_persons=1, fps=15.0, summary_state=sparse_state)
        self.assertEqual(out.shape, (480, 640, 3))

        # Person box with sparse state
        draw_person_box(frame, [50, 50, 150, 250], pid=5, state=sparse_state)
        self.assertTrue(np.any(frame > 0))

    # ─────────────────────────────────────────────────────────────
    # Visual Boundary & 3D Integration
    # ─────────────────────────────────────────────────────────────
    def test_3d_panel_visual_boundary(self):
        """Verifies 3D canvas boundary divider and HUD card rendering."""
        canvas = self.vis_3d.render(
            world_landmarks=None,
            activity_label="Standing",
            confidence=0.97,
            is_fall=False,
        )
        self.assertEqual(canvas.shape, (480, 320, 3))
        # Left 2 pixels should contain the border divider
        border_col1 = canvas[100, 0]
        self.assertTrue(np.all(border_col1 == (42, 38, 52)))

        # Stack test (dual-panel 2D + 3D)
        frame_2d = np.zeros((480, 640, 3), dtype=np.uint8)
        dual = np.hstack([frame_2d, canvas])
        self.assertEqual(dual.shape, (480, 960, 3))


if __name__ == "__main__":
    unittest.main()

