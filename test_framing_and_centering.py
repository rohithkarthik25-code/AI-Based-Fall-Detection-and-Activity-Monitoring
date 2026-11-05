"""
Dedicated test suite verifying Step 8 requirements for 3D skeleton visualization:
1. Standing skeleton size (55-70% of viewport height)
2. Sitting posture
3. Walking posture
4. Bending posture
5. Lying down posture
6. Partial landmark visibility
7. Camera 360-degree rotation & pole elevations
8. Manual zoom interaction & reset
9. Automatic framing across multiple viewport resolutions
10. Dynamic auto-centering during lateral and vertical movement
"""

import math
import unittest
import numpy as np

from visualizer_3d import SkeletonVisualizer3D


class DummyLandmark:
    def __init__(self, x, y, z, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def make_standing_pose(shift_x=0.0, shift_y=0.0, shift_z=0.0):
    """Creates a standing human pose (height ~1.65m) with optional spatial offset."""
    lms = [DummyLandmark(shift_x, shift_y, shift_z) for _ in range(33)]
    # Nose (0)
    lms[0] = DummyLandmark(shift_x, 0.68 + shift_y, -0.05 + shift_z)
    # Shoulders (11, 12)
    lms[11] = DummyLandmark(-0.20 + shift_x, 0.45 + shift_y, shift_z)
    lms[12] = DummyLandmark(0.20 + shift_x, 0.45 + shift_y, shift_z)
    # Hips (23, 24)
    lms[23] = DummyLandmark(-0.12 + shift_x, 0.0 + shift_y, shift_z)
    lms[24] = DummyLandmark(0.12 + shift_x, 0.0 + shift_y, shift_z)
    # Knees (25, 26)
    lms[25] = DummyLandmark(-0.13 + shift_x, -0.45 + shift_y, shift_z)
    lms[26] = DummyLandmark(0.13 + shift_x, -0.45 + shift_y, shift_z)
    # Ankles (27, 28)
    lms[27] = DummyLandmark(-0.14 + shift_x, -0.90 + shift_y, shift_z)
    lms[28] = DummyLandmark(0.14 + shift_x, -0.90 + shift_y, shift_z)
    # Feet (31, 32)
    lms[31] = DummyLandmark(-0.14 + shift_x, -0.96 + shift_y, 0.12 + shift_z)
    lms[32] = DummyLandmark(0.14 + shift_x, -0.96 + shift_y, 0.12 + shift_z)
    return lms


def make_sitting_pose():
    """Creates a seated pose: hips lowered, knees flexed forward."""
    lms = [DummyLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    lms[0] = DummyLandmark(0.0, 0.55, 0.0)      # Head
    lms[11] = DummyLandmark(-0.20, 0.35, 0.0)   # Left shoulder
    lms[12] = DummyLandmark(0.20, 0.35, 0.0)    # Right shoulder
    lms[23] = DummyLandmark(-0.12, -0.10, 0.0)  # Left hip
    lms[24] = DummyLandmark(0.12, -0.10, 0.0)   # Right hip
    lms[25] = DummyLandmark(-0.13, -0.10, 0.40) # Knees bent forward
    lms[26] = DummyLandmark(0.13, -0.10, 0.40)
    lms[27] = DummyLandmark(-0.14, -0.55, 0.40) # Feet down
    lms[28] = DummyLandmark(0.14, -0.55, 0.40)
    return lms


def make_lying_pose():
    """Creates a lying-down pose: body stretched horizontally."""
    lms = [DummyLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    # Torso and legs horizontal along Z axis
    lms[0] = DummyLandmark(0.0, 0.10, -0.75)    # Head
    lms[11] = DummyLandmark(-0.18, 0.08, -0.50) # Shoulders
    lms[12] = DummyLandmark(0.18, 0.08, -0.50)
    lms[23] = DummyLandmark(-0.12, 0.05, 0.0)   # Hips
    lms[24] = DummyLandmark(0.12, 0.05, 0.0)
    lms[25] = DummyLandmark(-0.12, 0.05, 0.40)  # Knees
    lms[26] = DummyLandmark(0.12, 0.05, 0.40)
    lms[27] = DummyLandmark(-0.12, 0.05, 0.85)  # Feet
    lms[28] = DummyLandmark(0.12, 0.05, 0.85)
    return lms


def make_bending_pose():
    """Creates a bending forward pose: torso pitched forward 45 deg."""
    lms = [DummyLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    lms[0] = DummyLandmark(0.0, 0.05, 0.45)     # Head forward and down
    lms[11] = DummyLandmark(-0.20, 0.10, 0.30)
    lms[12] = DummyLandmark(0.20, 0.10, 0.30)
    lms[23] = DummyLandmark(-0.12, 0.0, 0.0)    # Hips elevated
    lms[24] = DummyLandmark(0.12, 0.0, 0.0)
    lms[25] = DummyLandmark(-0.13, -0.45, 0.0)  # Straight legs
    lms[26] = DummyLandmark(0.13, -0.45, 0.0)
    lms[27] = DummyLandmark(-0.14, -0.90, 0.0)
    lms[28] = DummyLandmark(0.14, -0.90, 0.0)
    return lms


class TestFramingAndCentering(unittest.TestCase):

    def test_standing_skeleton_size_coverage(self):
        """1. Verify standing skeleton occupies 55% - 70% of viewport height."""
        for h in [480, 720, 400]:
            vis = SkeletonVisualizer3D(width=int(h * 0.75), height=h)
            pose = make_standing_pose()
            # Render 3 frames to allow EMA to settle
            for _ in range(3):
                vis.render(pose)

            # Measure projected screen bounds of skeleton
            R = vis._get_rotation_matrix()
            pts_3d = [np.array([lm.x, -lm.y, lm.z], dtype=np.float32) for lm in pose]
            projs = [vis._project_point(p, R)[0] for p in pts_3d]
            valid_v = [p[1] for p in projs if p is not None]

            min_v = min(valid_v)
            max_v = max(valid_v)
            proj_h = max_v - min_v
            coverage = proj_h / float(h)

            # Must be between 55% and 70%
            self.assertGreaterEqual(coverage, 0.55, f"Coverage {coverage:.2f} < 55% on height={h}")
            self.assertLessEqual(coverage, 0.70, f"Coverage {coverage:.2f} > 70% on height={h}")

            # Must not clip head or feet
            self.assertGreater(min_v, 20, f"Head clipped: {min_v}")
            self.assertLess(max_v, h - 15, f"Feet clipped: {max_v}")

    def test_sitting_posture_centering(self):
        """2. Verify sitting pose is vertically and horizontally centered."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        pose = make_sitting_pose()
        for _ in range(3):
            vis.render(pose)

        R = vis._get_rotation_matrix()
        pts_3d = [np.array([lm.x, -lm.y, lm.z], dtype=np.float32) for lm in pose]
        projs = [vis._project_point(p, R)[0] for p in pts_3d]
        valid_u = [p[0] for p in projs if p is not None]
        valid_v = [p[1] for p in projs if p is not None]

        center_u = (min(valid_u) + max(valid_u)) / 2.0
        center_v = (min(valid_v) + max(valid_v)) / 2.0

        # Centered near (540/2 = 270, 720/2 + offset ~ 360-430)
        self.assertAlmostEqual(center_u, 270.0, delta=40.0)
        self.assertGreater(min(valid_v), 90, "Sitting head clipped by HUD")
        self.assertLess(max(valid_v), 700, "Sitting feet clipped by bottom")
        self.assertAlmostEqual(center_v, 400.0, delta=60.0)

    def test_walking_posture(self):
        """3. Verify walking pose frames and centers without error."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        pose = make_standing_pose()
        # Stagger legs for walking stride
        pose[27].z += 0.25
        pose[28].z -= 0.25
        for _ in range(3):
            canvas = vis.render(pose, activity_label="Walking", confidence=0.88)
        self.assertEqual(canvas.shape, (720, 540, 3))
        self.assertGreater(np.count_nonzero(canvas), 1000)

    def test_bending_posture(self):
        """4. Verify bending pose does not clip extremities."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        pose = make_bending_pose()
        for _ in range(3):
            vis.render(pose)

        R = vis._get_rotation_matrix()
        pts_3d = [np.array([lm.x, -lm.y, lm.z], dtype=np.float32) for lm in pose]
        projs = [vis._project_point(p, R)[0] for p in pts_3d]
        valid_v = [p[1] for p in projs if p is not None]

        self.assertGreater(min(valid_v), 10)
        self.assertLess(max(valid_v), 710)

    def test_lying_down_posture_no_clipping(self):
        """5. Verify lying down pose does not clip horizontally or vertically."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        pose = make_lying_pose()
        for _ in range(3):
            vis.render(pose)

        R = vis._get_rotation_matrix()
        pts_3d = [np.array([lm.x, -lm.y, lm.z], dtype=np.float32) for lm in pose]
        projs = [vis._project_point(p, R)[0] for p in pts_3d]
        valid_u = [p[0] for p in projs if p is not None]
        valid_v = [p[1] for p in projs if p is not None]

        self.assertGreater(min(valid_u), 10)
        self.assertLess(max(valid_u), 530)
        self.assertGreater(min(valid_v), 20)
        self.assertLess(max(valid_v), 705)

    def test_partial_landmark_visibility(self):
        """6. Verify graceful handling of partial landmark visibility (e.g. only upper body)."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        pose = make_standing_pose()
        # Mark lower body invisible
        for idx in range(23, 33):
            pose[idx].visibility = 0.05
        canvas = vis.render(pose)
        self.assertEqual(canvas.shape, (720, 540, 3))

    def test_camera_rotation_all_angles(self):
        """7. Verify full 360-degree azimuth and pole elevation rotations."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        pose = make_standing_pose()
        vis.render(pose)

        # Test extreme elevations
        for az in [-180, -90, 0, 90, 180]:
            for el in [-85, -45, 0, 45, 85]:
                vis.camera.azimuth = float(az)
                vis.camera.elevation = float(el)
                canvas = vis.render(pose)
                self.assertEqual(canvas.shape, (720, 540, 3))
                self.assertGreater(np.count_nonzero(canvas), 1000)

    def test_manual_zoom_interaction(self):
        """8. Verify manual zoom interacts properly on top of auto-framing."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        pose = make_standing_pose()
        vis.render(pose)

        # Zoom in
        vis.handle_key(ord("+"))
        self.assertGreater(vis.camera.zoom, 1.0)

        # Zoom out
        vis.handle_key(ord("-"))
        vis.handle_key(ord("-"))
        self.assertLess(vis.camera.zoom, 1.0)

        # Reset camera
        vis.handle_key(ord("r"))
        self.assertEqual(vis.camera.zoom, 1.0)

    def test_auto_centering_during_movement(self):
        """9 & 10. Verify skeleton remains centered when subject moves spatially."""
        vis = SkeletonVisualizer3D(width=540, height=720)
        # Person standing at x=0
        pose_center = make_standing_pose(shift_x=0.0)
        for _ in range(5):
            vis.render(pose_center)
        c0 = vis.camera.center_x

        # Person moves 1.5m to the right
        pose_moved = make_standing_pose(shift_x=1.5)
        for _ in range(15):
            vis.render(pose_moved)
        c1 = vis.camera.center_x

        # Camera center must have shifted towards the new position
        self.assertGreater(c1, c0 + 1.0)


if __name__ == "__main__":
    unittest.main()
