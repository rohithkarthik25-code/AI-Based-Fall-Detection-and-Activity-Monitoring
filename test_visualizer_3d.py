"""
Unit test suite for SkeletonVisualizer3D.
Validates 3D transformations, perspective projections, mouse interaction simulation,
depth-sorting, and rendering integrity.
"""

import math
import unittest
import numpy as np
import cv2

from visualizer_3d import SkeletonVisualizer3D, CameraState


class DummyLandmark:
    def __init__(self, x, y, z, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class TestSkeletonVisualizer3D(unittest.TestCase):

    def setUp(self):
        self.vis = SkeletonVisualizer3D(width=400, height=400, enable_grid=True)

    def _create_synthetic_pose(self):
        """Creates a realistic standing human skeleton in 3D meters (hip-centered)."""
        lms = [DummyLandmark(0, 0, 0) for _ in range(33)]
        # Head / Nose
        lms[0] = DummyLandmark(0.0, 0.65, -0.05)
        # Shoulders (11, 12)
        lms[11] = DummyLandmark(-0.20, 0.45, 0.0)
        lms[12] = DummyLandmark(0.20, 0.45, 0.0)
        # Elbows (13, 14)
        lms[13] = DummyLandmark(-0.30, 0.20, 0.0)
        lms[14] = DummyLandmark(0.30, 0.20, 0.0)
        # Wrists (15, 16)
        lms[15] = DummyLandmark(-0.35, -0.05, 0.0)
        lms[16] = DummyLandmark(0.35, -0.05, 0.0)
        # Hips (23, 24)
        lms[23] = DummyLandmark(-0.12, 0.0, 0.0)
        lms[24] = DummyLandmark(0.12, 0.0, 0.0)
        # Knees (25, 26)
        lms[25] = DummyLandmark(-0.13, -0.45, 0.0)
        lms[26] = DummyLandmark(0.13, -0.45, 0.0)
        # Ankles (27, 28)
        lms[27] = DummyLandmark(-0.14, -0.90, 0.0)
        lms[28] = DummyLandmark(0.14, -0.90, 0.0)
        # Feet (31, 32)
        lms[31] = DummyLandmark(-0.14, -0.95, 0.12)
        lms[32] = DummyLandmark(0.14, -0.95, 0.12)

        return lms

    def test_rotation_matrix_orthogonality(self):
        """Test that rotation matrix is a valid SO(3) orthogonal matrix."""
        R = self.vis._get_rotation_matrix()
        # R @ R.T should be identity
        ident = R @ R.T
        np.testing.assert_allclose(ident, np.eye(3), atol=1e-5)
        # Determinant must be +1
        det = np.linalg.det(R)
        self.assertAlmostEqual(det, 1.0, places=5)

    def test_projection_center(self):
        """Origin (0,0,0) at elevation 0, azimuth 0 should project to screen center."""
        self.vis.camera.azimuth = 0.0
        self.vis.camera.elevation = 0.0
        self.vis.camera.center_y_offset = 0.0
        R = self.vis._get_rotation_matrix()

        pt = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        proj, z = self.vis._project_point(pt, R, cam_dist=2.0)

        self.assertIsNotNone(proj)
        self.assertAlmostEqual(proj[0], self.vis.width / 2.0, delta=1.0)
        self.assertAlmostEqual(proj[1], self.vis.height / 2.0, delta=1.0)
        self.assertAlmostEqual(z, 2.0, places=2)

    def test_mouse_orbit_rotation(self):
        """Simulate mouse click and drag to rotate camera."""
        initial_az = self.vis.camera.azimuth
        initial_el = self.vis.camera.elevation

        # 1. Left button down inside viewport
        captured = self.vis.handle_mouse(cv2.EVENT_LBUTTONDOWN, 200, 200)
        self.assertTrue(captured)
        self.assertTrue(self.vis.is_dragging)

        # 2. Mouse move (drag right +40px, up -20px)
        self.vis.handle_mouse(cv2.EVENT_MOUSEMOVE, 240, 180)
        self.assertNotEqual(self.vis.camera.azimuth, initial_az)
        self.assertNotEqual(self.vis.camera.elevation, initial_el)

        # 3. Left button up
        self.vis.handle_mouse(cv2.EVENT_LBUTTONUP, 240, 180)
        self.assertFalse(self.vis.is_dragging)

    def test_mouse_wheel_and_zoom(self):
        """Test zoom in and zoom out via mouse wheel and right-drag."""
        init_zoom = self.vis.camera.zoom
        # Zoom in via mouse wheel
        self.vis.handle_mouse(10, 200, 200, flags=1) # scroll up
        self.assertGreater(self.vis.camera.zoom, init_zoom)

        # Double click to reset
        self.vis.handle_mouse(cv2.EVENT_LBUTTONDBLCLK, 200, 200)
        self.assertAlmostEqual(self.vis.camera.zoom, 1.0)

    def test_presets(self):
        """Test preset views."""
        self.vis.set_preset_view("front")
        self.assertEqual(self.vis.camera.azimuth, 0.0)
        self.assertEqual(self.vis.camera.elevation, 0.0)

        self.vis.set_preset_view("top")
        self.assertEqual(self.vis.camera.elevation, 85.0)

        self.vis.set_preset_view("iso")
        self.assertEqual(self.vis.camera.azimuth, 25.0)
        self.assertEqual(self.vis.camera.elevation, 12.0)

    def test_full_render_pipeline(self):
        """Test that full render output is a valid BGR image with correct dimensions."""
        pose = self._create_synthetic_pose()
        canvas = self.vis.render(
            world_landmarks=pose,
            activity_label="Standing",
            confidence=0.92,
            is_fall=False,
        )

        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape, (self.vis.height, self.vis.width, 3))
        self.assertEqual(canvas.dtype, np.uint8)

        # Ensure canvas is not blank
        self.assertGreater(np.count_nonzero(canvas), 1000)

    def test_nan_and_inf_handling(self):
        """Test that NaN and Inf coordinates do not crash the projection or renderer."""
        R = self.vis._get_rotation_matrix()
        proj, z = self.vis._project_point(np.array([float("nan"), 0.0, 0.0], dtype=np.float32), R)
        self.assertIsNone(proj)

        proj_inf, _ = self.vis._project_point(np.array([0.0, float("inf"), 0.0], dtype=np.float32), R)
        self.assertIsNone(proj_inf)

        # Pose with some NaN landmarks
        pose = self._create_synthetic_pose()
        pose[0] = DummyLandmark(float("nan"), float("nan"), float("nan"))
        pose[11] = DummyLandmark(float("inf"), 0.0, 0.0)
        canvas = self.vis.render(pose)
        self.assertEqual(canvas.shape, (self.vis.height, self.vis.width, 3))

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcut handler."""
        # 'f' -> front
        self.assertTrue(self.vis.handle_key(ord("f")))
        self.assertEqual(self.vis.camera.azimuth, 0.0)

        # 's' -> side
        self.assertTrue(self.vis.handle_key(ord("s")))
        self.assertEqual(self.vis.camera.azimuth, 90.0)

        # 't' -> top
        self.assertTrue(self.vis.handle_key(ord("t")))
        self.assertEqual(self.vis.camera.elevation, 85.0)

        # 'r' -> reset
        self.assertTrue(self.vis.handle_key(ord("r")))
        self.assertEqual(self.vis.camera.azimuth, 25.0)
        self.assertEqual(self.vis.camera.elevation, 12.0)

        # '+' -> zoom in
        z_before = self.vis.camera.zoom
        self.assertTrue(self.vis.handle_key(ord("+")))
        self.assertGreater(self.vis.camera.zoom, z_before)

        # '-' -> zoom out
        self.assertTrue(self.vis.handle_key(ord("-")))

        # unknown key -> False
        self.assertFalse(self.vis.handle_key(ord("x")))

    def test_mouse_outside_viewport(self):
        """Test that mouse clicks outside the 3D viewport return False and do nothing."""
        # Click at (-50, 100) or (500, 100) when width=400
        captured = self.vis.handle_mouse(cv2.EVENT_LBUTTONDOWN, 500, 100)
        self.assertFalse(captured)
        self.assertFalse(self.vis.is_dragging)


if __name__ == "__main__":
    unittest.main()

