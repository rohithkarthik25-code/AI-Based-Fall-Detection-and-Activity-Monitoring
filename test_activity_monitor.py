"""
Comprehensive unit test suite for ActivityMonitor.
Tests all biomechanical postures, debouncing, 3D/2D angles, duration tracking,
and fall-override safety.
"""

import unittest
from types import SimpleNamespace
from activity_monitor import ActivityMonitor, ActivityType


class TestActivityMonitor(unittest.TestCase):

    def setUp(self):
        self.monitor = ActivityMonitor(debounce_frames=3, default_fps=25.0)

    def _make_dummy_features(self, torso_angle=10.0, aspect_ratio=0.4, torso_tilt_3d=10.0, height_drop=0.0):
        return SimpleNamespace(
            torso_angle=torso_angle,
            aspect_ratio=aspect_ratio,
            torso_tilt_3d=torso_tilt_3d,
            height_drop=height_drop,
            post_impact_stillness=0.0,
        )

    def _make_dummy_pose(self, knee_angle=175.0):
        # 3D points forming a specific knee angle
        # Hip=(0, 0, 0), Knee=(0, -1, 0), Ankle depends on angle
        import math
        rad = math.radians(180 - knee_angle)
        # Ankle relative to knee
        ax = math.sin(rad)
        ay = -1.0 - math.cos(rad)
        
        class Pt:
            def __init__(self, x, y, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        wl = [Pt(0, 0, 0) for _ in range(33)]
        # Left leg: Hip=23, Knee=25, Ankle=27
        wl[23] = Pt(0, 0, 0)
        wl[25] = Pt(0, -1.0, 0)
        wl[27] = Pt(ax, ay, 0)
        # Right leg: Hip=24, Knee=26, Ankle=28
        wl[24] = Pt(0, 0, 0)
        wl[26] = Pt(0, -1.0, 0)
        wl[28] = Pt(ax, ay, 0)

        return SimpleNamespace(
            has_pose=True,
            landmarks=None,
            world_landmarks=wl,
            visibility=0.9,
            crop_shape=(100, 100, 3)
        )

    def test_standing(self):
        state = {}
        box = (100, 50, 150, 200) # tall box (w=50, h=150)
        pose = self._make_dummy_pose(knee_angle=175.0)
        feat = self._make_dummy_features(torso_angle=8.0, aspect_ratio=0.33)

        # Feed 3 frames for debounce
        for _ in range(3):
            metrics = self.monitor.update(state, box, pose, feat, dt=0.1)

        self.assertEqual(metrics.activity, ActivityType.STANDING.value)
        self.assertGreater(metrics.avg_knee_angle, 160.0)
        self.assertAlmostEqual(state["activity_durations"][ActivityType.STANDING.value], 0.3)

    def test_sitting(self):
        state = {}
        box = (100, 100, 180, 200) # squarer box (w=80, h=100)
        pose = self._make_dummy_pose(knee_angle=95.0) # flexed knees ~95°
        feat = self._make_dummy_features(torso_angle=12.0, aspect_ratio=0.8, height_drop=0.25)

        for _ in range(3):
            metrics = self.monitor.update(state, box, pose, feat, dt=0.1)

        self.assertEqual(metrics.activity, ActivityType.SITTING.value)
        self.assertLess(metrics.avg_knee_angle, 120.0)

    def test_bending(self):
        state = {}
        box = (100, 80, 160, 200)
        pose = self._make_dummy_pose(knee_angle=170.0) # straight knees
        feat = self._make_dummy_features(torso_angle=48.0, aspect_ratio=0.55) # leaning torso

        for _ in range(3):
            metrics = self.monitor.update(state, box, pose, feat, dt=0.1)

        self.assertEqual(metrics.activity, ActivityType.BENDING.value)

    def test_lying_down(self):
        state = {}
        box = (50, 150, 250, 210) # wide box on floor (w=200, h=60)
        pose = self._make_dummy_pose(knee_angle=165.0)
        feat = self._make_dummy_features(torso_angle=75.0, aspect_ratio=3.3) # horizontal torso

        for _ in range(3):
            metrics = self.monitor.update(state, box, pose, feat, dt=0.1)

        self.assertEqual(metrics.activity, ActivityType.LYING_DOWN.value)

    def test_walking(self):
        state = {}
        pose = self._make_dummy_pose(knee_angle=168.0)
        feat = self._make_dummy_features(torso_angle=10.0, aspect_ratio=0.4)

        # Move box horizontally each frame to simulate walking speed
        x = 100
        for _ in range(5):
            box = (x, 50, x + 50, 200)
            metrics = self.monitor.update(state, box, pose, feat, dt=0.04)
            x += 12 # significant horizontal motion

        self.assertEqual(metrics.activity, ActivityType.WALKING.value)
        self.assertGreater(metrics.horizontal_speed, self.monitor.walk_speed_threshold)

    def test_fall_override(self):
        state = {"fall_detected": True}
        box = (50, 150, 250, 210)
        pose = self._make_dummy_pose(knee_angle=120.0)
        feat = self._make_dummy_features(torso_angle=80.0, aspect_ratio=2.5)

        # Even on frame 1 without debouncing, fall should trigger immediately!
        metrics = self.monitor.update(state, box, pose, feat, dt=0.1)
        self.assertEqual(metrics.activity, ActivityType.FALLEN.value)

    def test_duration_and_percentages(self):
        state = {}
        box = (100, 50, 150, 200)
        pose_stand = self._make_dummy_pose(knee_angle=175.0)
        feat_stand = self._make_dummy_features(torso_angle=8.0, aspect_ratio=0.33)

        pose_sit = self._make_dummy_pose(knee_angle=90.0)
        feat_sit = self._make_dummy_features(torso_angle=10.0, aspect_ratio=0.8)

        # 5 frames standing (dt=0.1s -> 0.5s)
        for _ in range(5):
            self.monitor.update(state, box, pose_stand, feat_stand, dt=0.1)

        # 5 frames sitting (dt=0.1s -> 0.5s)
        for _ in range(5):
            metrics = self.monitor.update(state, box, pose_sit, feat_sit, dt=0.1)

        total_time = state["total_tracked_time"]
        self.assertAlmostEqual(total_time, 1.0)
        self.assertIn("Standing", metrics.time_percentages)
        self.assertIn("Sitting", metrics.time_percentages)
        self.assertGreater(state["posture_transitions"], 0)


if __name__ == "__main__":
    unittest.main()

