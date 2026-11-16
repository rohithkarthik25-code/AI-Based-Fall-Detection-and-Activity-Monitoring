"""
Unit and Edge-Case Tests for 3D Viewport Biomechanics Engine (Option A + Option C).
Tests Center of Mass (CoM), Base of Support (BoS), Stability Margins,
Motion Trajectory Velocity Surges, and Edge Cases (NaN, Inf, Occlusion, Degenerate Polygons).
"""

import math
import unittest
import numpy as np

from visualizer_3d import BiomechanicsEngine, SkeletonVisualizer3D


class DummyLandmark:
    """Mock landmark for 3D pose tests."""
    def __init__(self, x: float, y: float, z: float, visibility: float = 0.9):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)


class TestViewportBiomechanics(unittest.TestCase):

    def setUp(self):
        self.engine = BiomechanicsEngine(history_len=10)

    def _create_upright_pose(self):
        """Creates a standardized upright human skeleton in metric meters (hip-centered)."""
        lms = [DummyLandmark(0.0, 0.0, 0.0) for _ in range(33)]
        # Head (0)
        lms[0] = DummyLandmark(0.0, 0.70, 0.0)
        # Shoulders (11, 12)
        lms[11] = DummyLandmark(-0.20, 0.50, 0.0)
        lms[12] = DummyLandmark(0.20, 0.50, 0.0)
        # Elbows (13, 14)
        lms[13] = DummyLandmark(-0.30, 0.25, 0.0)
        lms[14] = DummyLandmark(0.30, 0.25, 0.0)
        # Wrists (15, 16)
        lms[15] = DummyLandmark(-0.32, 0.0, 0.0)
        lms[16] = DummyLandmark(0.32, 0.0, 0.0)
        # Hips (23, 24)
        lms[23] = DummyLandmark(-0.15, 0.0, 0.0)
        lms[24] = DummyLandmark(0.15, 0.0, 0.0)
        # Knees (25, 26)
        lms[25] = DummyLandmark(-0.15, -0.45, 0.0)
        lms[26] = DummyLandmark(0.15, -0.45, 0.0)
        # Ankles (27, 28)
        lms[27] = DummyLandmark(-0.15, -0.90, 0.0)
        lms[28] = DummyLandmark(0.15, -0.90, 0.0)
        # Heels (29, 30)
        lms[29] = DummyLandmark(-0.15, -0.92, -0.08)
        lms[30] = DummyLandmark(0.15, -0.92, -0.08)
        # Toes (31, 32)
        lms[31] = DummyLandmark(-0.15, -0.94, 0.14)
        lms[32] = DummyLandmark(0.15, -0.94, 0.14)
        return lms

    def test_center_of_mass_upright_standing(self):
        """CoM for upright standing should be horizontally centered with positive vertical elevation above hips."""
        pose = self._create_upright_pose()
        com = BiomechanicsEngine.compute_center_of_mass(pose)
        self.assertIsNotNone(com)
        self.assertAlmostEqual(com[0], 0.0, delta=0.05, msg="CoM X should be near zero for symmetric upright stance")
        self.assertAlmostEqual(com[2], 0.0, delta=0.05, msg="CoM Z should be near zero for planar upright stance")
        # In hip-centered coordinates (hips at y=0, shoulders at y=0.5, ankles at y=-0.9):
        # 0.50*0 + 0.35*0.50 + 0.10*(-0.90) + 0.05*0.70 = 0.175 - 0.090 + 0.035 = 0.12m
        self.assertGreater(com[1], 0.05)
        self.assertLess(com[1], 0.30)

    def test_center_of_mass_lateral_fall_tilt(self):
        """When torso tilts severely to the right (+X), CoM X shifts decisively in the tilt direction."""
        pose = self._create_upright_pose()
        # Severe right tilt: head and shoulders lean right by +0.45m
        pose[0].x += 0.45
        pose[11].x += 0.40
        pose[12].x += 0.40

        com_tilted = BiomechanicsEngine.compute_center_of_mass(pose)
        self.assertIsNotNone(com_tilted)
        self.assertGreater(com_tilted[0], 0.12, msg="CoM should shift significantly to the right under tilt")

    def test_base_of_support_polygon_upright(self):
        """Base of support for two feet should form a convex polygon spanning left and right feet."""
        pose = self._create_upright_pose()
        bos = BiomechanicsEngine.compute_base_of_support_polygon(pose)
        self.assertIsNotNone(bos)
        self.assertGreaterEqual(len(bos), 3, "BoS polygon must have at least 3 vertices")

        # Minimum X should be on left foot (around -0.15m), max X on right foot (+0.15m)
        min_x = np.min(bos[:, 0])
        max_x = np.max(bos[:, 0])
        self.assertLessEqual(min_x, -0.14)
        self.assertGreaterEqual(max_x, 0.14)

    def test_point_in_polygon_containment(self):
        """Test ray-casting 2D point-in-polygon containment logic."""
        # Square from (-1, -1) to (1, 1)
        poly = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)

        # Inside points
        self.assertTrue(BiomechanicsEngine.is_point_in_polygon(0.0, 0.0, poly))
        self.assertTrue(BiomechanicsEngine.is_point_in_polygon(0.5, -0.5, poly))

        # Outside points
        self.assertFalse(BiomechanicsEngine.is_point_in_polygon(2.0, 0.0, poly))
        self.assertFalse(BiomechanicsEngine.is_point_in_polygon(-1.5, 0.0, poly))
        self.assertFalse(BiomechanicsEngine.is_point_in_polygon(0.0, 1.5, poly))

    def test_stability_margin_positive_for_stable_standing(self):
        """Upright stance has CoM safely inside BoS with positive stability margin."""
        pose = self._create_upright_pose()
        result = self.engine.evaluate_stability(pose, timestamp=1.0)
        self.assertTrue(result["is_stable"])
        self.assertEqual(result["status"], "STABLE")
        self.assertGreater(result["margin_m"], 0.05, "Stability margin must be positive inside BoS")

    def test_stability_margin_negative_for_unstable_lean(self):
        """When leaning far outside the base of support, stability margin must be negative."""
        pose = self._create_upright_pose()
        # Shift entire upper body and hips far out to X = +0.8m
        for idx in [0, 11, 12, 13, 14, 23, 24]:
            pose[idx].x += 0.85

        result = self.engine.evaluate_stability(pose, timestamp=2.0)
        self.assertFalse(result["is_stable"])
        self.assertEqual(result["status"], "UNSTABLE")
        self.assertLess(result["margin_m"], -0.10, "Stability margin must be negative when CoM exits BoS")

    def test_motion_trajectory_fifo_and_velocity(self):
        """Trajectory buffer caps at history_len and correctly calculates descent speed."""
        engine = BiomechanicsEngine(history_len=5)
        com = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Add 8 positions (should cap at 5)
        for i in range(8):
            y_pos = 1.0 - i * 0.1  # descending by 0.1m every 0.1s => 1.0 m/s
            t = i * 0.1
            v = engine.update_trajectory(np.array([0.0, y_pos, 0.0]), timestamp=t)

        self.assertEqual(len(engine.trajectory_history), 5)
        self.assertAlmostEqual(v, 1.0, delta=0.05, msg="Descent velocity should be ~1.0 m/s")

    def test_edge_case_nan_and_inf_landmarks(self):
        """NaN and Inf in landmarks must not crash CoM or BoS calculations."""
        pose = self._create_upright_pose()
        pose[0] = DummyLandmark(float("nan"), 0.0, 0.0)
        pose[11] = DummyLandmark(0.0, float("inf"), 0.0)
        pose[27] = DummyLandmark(float("-inf"), 0.0, 0.0)

        # Should handle gracefully without raising exception
        com = BiomechanicsEngine.compute_center_of_mass(pose)
        self.assertIsNotNone(com)
        self.assertFalse(np.isnan(com).any())
        self.assertFalse(np.isinf(com).any())

        bos = BiomechanicsEngine.compute_base_of_support_polygon(pose)
        self.assertIsNotNone(bos)

    def test_edge_case_none_and_empty_inputs(self):
        """Empty or None landmark lists return safe fallbacks."""
        self.assertIsNone(BiomechanicsEngine.compute_center_of_mass(None))
        self.assertIsNone(BiomechanicsEngine.compute_center_of_mass([]))
        self.assertIsNone(BiomechanicsEngine.compute_base_of_support_polygon(None))
        self.assertIsNone(BiomechanicsEngine.compute_base_of_support_polygon([]))

        eval_none = self.engine.evaluate_stability(None)
        self.assertEqual(eval_none["status"], "UNKNOWN")
        self.assertEqual(eval_none["margin_m"], 0.0)

    def test_edge_case_coincident_feet_expansion(self):
        """When feet are coincident (e.g. feet at exact same point), expands to safe minimal polygon."""
        pose = self._create_upright_pose()
        # Set all feet points to (0, 0)
        for idx in [27, 28, 29, 30, 31, 32]:
            pose[idx].x = 0.0
            pose[idx].z = 0.0

        bos = BiomechanicsEngine.compute_base_of_support_polygon(pose)
        self.assertIsNotNone(bos)
        self.assertGreaterEqual(len(bos), 3)
        # Margin calculation against coincident foot should not divide by zero
        com = np.array([0.0, 0.5, 0.0])
        margin = BiomechanicsEngine.compute_stability_margin(com, bos)
        self.assertFalse(math.isnan(margin))
        self.assertFalse(math.isinf(margin))

    def test_edge_case_zero_dt_in_trajectory(self):
        """dt = 0.0 in trajectory update should not trigger ZeroDivisionError."""
        engine = BiomechanicsEngine()
        com = np.array([0.0, 1.0, 0.0])
        v1 = engine.update_trajectory(com, timestamp=1.0)
        v2 = engine.update_trajectory(com, timestamp=1.0)  # identical timestamp
        self.assertEqual(v2, 0.0)

    def test_skeleton_visualizer_integration(self):
        """SkeletonVisualizer3D has active biomechanics engine and integrates seamlessly."""
        vis = SkeletonVisualizer3D(width=400, height=400)
        self.assertTrue(hasattr(vis, "biomechanics"))
        pose = self._create_upright_pose()
        res = vis.biomechanics.evaluate_stability(pose, timestamp=0.5)
        self.assertIn("status", res)
        self.assertIn("margin_m", res)
        self.assertIn("velocity_mps", res)


if __name__ == "__main__":
    unittest.main()
