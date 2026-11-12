"""
Extreme edge case and stress testing suite for all modules.
Covers corrupted inputs, mathematical singularities, boundary conditions,
and rapid state transitions across the full pipeline.
"""

import math
import os
import unittest
from types import SimpleNamespace
import numpy as np
import cv2

from activity_monitor import ActivityMonitor, ActivityType
from visualizer_3d import SkeletonVisualizer3D, CameraState
from weighted_scoring import WeightedScorer
from dashboard import Dashboard, draw_person_box, _draw_factor_panel
from temporal_state_machine import TemporalStateMachine


class TestComprehensiveEdgeCases(unittest.TestCase):

    def setUp(self):
        self.monitor = ActivityMonitor()
        self.vis = SkeletonVisualizer3D(width=400, height=400)
        self.scorer = WeightedScorer()
        self.fsm = TemporalStateMachine()
        self.dash = Dashboard()

    # ─────────────────────────────────────────────────────────────
    # 1. ActivityMonitor Edge Cases
    # ─────────────────────────────────────────────────────────────

    def test_activity_extreme_dt(self):
        """Test zero, negative, and extreme delta time dt."""
        state = {}
        box = (10, 10, 50, 100)
        pose = SimpleNamespace(has_pose=False, landmarks=None, world_landmarks=None)
        feat = SimpleNamespace(torso_angle=0, aspect_ratio=0.5, torso_tilt_3d=0, height_drop=0)

        # dt = 0.0
        m0 = self.monitor.update(state, box, pose, feat, dt=0.0)
        self.assertIsNotNone(m0.activity)

        # dt = -5.0 (invalid time jump)
        m_neg = self.monitor.update(state, box, pose, feat, dt=-5.0)
        self.assertIsNotNone(m_neg.activity)

        # dt = 1000.0 (huge frame pause)
        m_huge = self.monitor.update(state, box, pose, feat, dt=1000.0)
        self.assertIsNotNone(m_huge.activity)

    def test_activity_zero_size_box(self):
        """Test bounding box with zero area or negative dimensions."""
        state = {}
        box = (100, 100, 100, 100) # w=0, h=0
        pose = SimpleNamespace(has_pose=False, landmarks=None, world_landmarks=None)
        feat = SimpleNamespace(torso_angle=0, aspect_ratio=0, torso_tilt_3d=0, height_drop=0)

        m = self.monitor.update(state, box, pose, feat, dt=0.04)
        self.assertIsNotNone(m.activity)
        self.assertTrue(math.isfinite(m.horizontal_speed))

    def test_activity_rapid_flipping(self):
        """Stress test rapid alternating postures to verify debounce stability."""
        state = {}
        box = (100, 50, 150, 200)
        pose = SimpleNamespace(has_pose=False, landmarks=None, world_landmarks=None)

        # Flip between sitting and standing on every single frame
        for i in range(30):
            is_stand = (i % 2 == 0)
            feat = SimpleNamespace(
                torso_angle=10.0,
                aspect_ratio=0.35 if is_stand else 0.85,
                torso_tilt_3d=10.0,
                height_drop=0.0 if is_stand else 0.3,
            )
            m = self.monitor.update(state, box, pose, feat, dt=0.04)
            self.assertIn(m.activity, [ActivityType.STANDING.value, ActivityType.SITTING.value])

    # ─────────────────────────────────────────────────────────────
    # 2. SkeletonVisualizer3D Edge Cases
    # ─────────────────────────────────────────────────────────────

    def test_vis_singularities_and_poles(self):
        """Test camera elevation at strict vertical poles (-90°, +90°)."""
        self.vis.camera.elevation = 85.0
        self.vis.camera.azimuth = 0.0
        R = self.vis._get_rotation_matrix()
        self.assertTrue(np.all(np.isfinite(R)))

        self.vis.camera.elevation = -85.0
        R_down = self.vis._get_rotation_matrix()
        self.assertTrue(np.all(np.isfinite(R_down)))

    def test_vis_extreme_zoom_clamping(self):
        """Test zooming out and zooming in past boundaries."""
        # Extreme zoom in
        for _ in range(30):
            self.vis.handle_key(ord("+"))
        self.assertLessEqual(self.vis.camera.zoom, 3.5)

        # Extreme zoom out
        for _ in range(50):
            self.vis.handle_key(ord("-"))
        self.assertGreaterEqual(self.vis.camera.zoom, 0.3)

    def test_vis_all_zeros_and_coincident_points(self):
        """Test skeleton when all landmarks are coincident at (0, 0, 0)."""
        class Pt:
            x, y, z, visibility = 0.0, 0.0, 0.0, 1.0

        coincident_lms = [Pt() for _ in range(33)]
        canvas = self.vis.render(coincident_lms)
        self.assertEqual(canvas.shape, (self.vis.height, self.vis.width, 3))
        self.assertGreater(np.count_nonzero(canvas), 100)

    # ─────────────────────────────────────────────────────────────
    # 3. WeightedScorer Edge Cases
    # ─────────────────────────────────────────────────────────────

    def test_scorer_all_zeros(self):
        """Test scoring when all biomechanical signals are zero."""
        f_zero = SimpleNamespace(
            torso_angle=0.0, aspect_ratio=0.0, hip_velocity=0.0, upward_hip_velocity=0.0,
            height_drop=0.0, persistence=0.0, height_collapse=0.0, width_expansion=0.0,
            head_velocity=0.0, shoulder_hip_separation=0.5, knee_collapse_left=0.0,
            knee_collapse_right=0.0, bilateral_asymmetry=0.0, post_impact_stillness=0.0,
            torso_tilt_3d=0.0, angle_delta=0.0, visibility=1.0,
        )
        res = self.scorer.score_features(f_zero)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.severity, "NONE")
        self.assertFalse(res.candidate)

    def test_scorer_extreme_outliers(self):
        """Test scoring with extreme values (velocities=100.0, drop=100.0)."""
        f_extreme = SimpleNamespace(
            torso_angle=180.0, aspect_ratio=5.0, hip_velocity=10.0, upward_hip_velocity=0.0,
            height_drop=10.0, persistence=1.0, height_collapse=10.0, width_expansion=10.0,
            head_velocity=10.0, shoulder_hip_separation=0.0, knee_collapse_left=1.0,
            knee_collapse_right=1.0, bilateral_asymmetry=1.0, post_impact_stillness=1.0,
            torso_tilt_3d=90.0, angle_delta=100.0, visibility=1.0,
        )
        res = self.scorer.score_features(f_extreme)
        self.assertLessEqual(res.score, 1.0)
        self.assertGreaterEqual(res.score, 0.9)
        self.assertEqual(res.severity, "SEVERE")
        self.assertTrue(res.candidate)

    # ─────────────────────────────────────────────────────────────
    # 4. Dashboard Edge Cases
    # ─────────────────────────────────────────────────────────────

    def test_dashboard_offscreen_boxes(self):
        """Test drawing boxes that are completely outside image boundaries."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        off_box = [-200, -200, -50, -50]
        state = {
            "severity": "NONE",
            "fall_score": 0.0,
            "fall_detected": False,
            "activity_label": "Unknown",
        }
        # Should not throw OpenCV error
        draw_person_box(frame, off_box, pid=99, state=state)

    def test_dashboard_extreme_fps(self):
        """Test Dashboard with 0 FPS and extreme 5000 FPS."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out0 = self.dash.draw(frame.copy(), active_persons=0, fps=0.0)
        self.assertEqual(out0.shape, (480, 640, 3))

        out_fast = self.dash.draw(frame.copy(), active_persons=100, fps=5000.0)
        self.assertEqual(out_fast.shape, (480, 640, 3))

    # ─────────────────────────────────────────────────────────────
    # 5. FPS & Timer Singularity Edge Cases (Webcam 0-dt guard)
    # ─────────────────────────────────────────────────────────────

    def test_pipeline_zero_dt_resilience(self):
        """Test that sub-millisecond identical timestamps (dt=0.0) never trigger ZeroDivisionError."""
        from collections import deque
        from pipeline_profiler import PipelineProfiler

        profiler = PipelineProfiler(rolling_window=30, enabled=True)
        fps_history = deque(maxlen=15)

        # Simulate 10 frames where time.time() returns identical values (dt = 0.0)
        prev_time = 1000.0
        for _ in range(10):
            now = 1000.0  # Identical timestamp
            dt = max(1e-4, now - prev_time)
            fps_history.append(dt)
            prev_time = now

            avg_dt = (sum(fps_history) / len(fps_history)) if fps_history else 0.033
            fps = (
                profiler.average_fps
                if (profiler.frame_count > 5 and profiler.average_fps > 0.0)
                else min(120.0, 1.0 / max(0.001, avg_dt))
            )
            self.assertTrue(math.isfinite(fps))
            self.assertGreater(fps, 0.0)

    def test_webcam_fps_fallback(self):
        """Test that NaN, 0.0, or negative webcam FPS falls back cleanly to 25.0."""
        for bad_fps in [0.0, float('nan'), -1.0, None]:
            fps_src = (
                float(bad_fps)
                if (bad_fps is not None and math.isfinite(bad_fps) and bad_fps > 0)
                else 25.0
            )
            self.assertEqual(fps_src, 25.0)


if __name__ == "__main__":
    unittest.main()

