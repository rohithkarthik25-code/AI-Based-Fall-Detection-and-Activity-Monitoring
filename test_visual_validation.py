"""
Visual validation test suite across all 6 postures:
1. Standing
2. Sitting
3. Walking
4. Bending
5. Lying down
6. Fallen (Emergency Red)
"""

import math
import os
import unittest
import numpy as np
import cv2

from visualizer_3d import SkeletonVisualizer3D, PALETTE


class DummyLandmark:
    def __init__(self, x, y, z, visibility=0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def create_posture(posture_type: str):
    lms = [DummyLandmark(0.0, 0.0, 0.0) for _ in range(33)]

    if posture_type == "standing":
        lms[0] = DummyLandmark(0.0, 0.68, -0.05)       # Head/Nose
        lms[2] = DummyLandmark(-0.035, 0.70, -0.04)    # L Eye
        lms[5] = DummyLandmark(0.035, 0.70, -0.04)     # R Eye
        lms[7] = DummyLandmark(-0.08, 0.69, 0.02)      # L Ear
        lms[8] = DummyLandmark(0.08, 0.69, 0.02)       # R Ear
        lms[11] = DummyLandmark(-0.20, 0.45, 0.0)      # L Shoulder
        lms[12] = DummyLandmark(0.20, 0.45, 0.0)       # R Shoulder
        lms[13] = DummyLandmark(-0.28, 0.20, 0.02)     # L Elbow
        lms[14] = DummyLandmark(0.28, 0.20, 0.02)      # R Elbow
        lms[15] = DummyLandmark(-0.32, -0.05, 0.04)    # L Wrist
        lms[16] = DummyLandmark(0.32, -0.05, 0.04)     # R Wrist
        lms[17] = DummyLandmark(-0.34, -0.12, 0.03)    # L Pinky
        lms[18] = DummyLandmark(0.34, -0.12, 0.03)     # R Pinky
        lms[19] = DummyLandmark(-0.32, -0.14, 0.04)    # L Index
        lms[20] = DummyLandmark(0.32, -0.14, 0.04)     # R Index
        lms[21] = DummyLandmark(-0.30, -0.10, 0.06)    # L Thumb
        lms[22] = DummyLandmark(0.30, -0.10, 0.06)     # R Thumb
        lms[23] = DummyLandmark(-0.12, 0.0, 0.0)       # L Hip
        lms[24] = DummyLandmark(0.12, 0.0, 0.0)        # R Hip
        lms[25] = DummyLandmark(-0.13, -0.45, 0.01)    # L Knee
        lms[26] = DummyLandmark(0.13, -0.45, 0.01)     # R Knee
        lms[27] = DummyLandmark(-0.14, -0.90, 0.0)     # L Ankle
        lms[28] = DummyLandmark(0.14, -0.90, 0.0)      # R Ankle
        lms[29] = DummyLandmark(-0.14, -0.96, -0.05)   # L Heel
        lms[30] = DummyLandmark(0.14, -0.96, -0.05)    # R Heel
        lms[31] = DummyLandmark(-0.14, -0.98, 0.12)    # L Toe
        lms[32] = DummyLandmark(0.14, -0.98, 0.12)     # R Toe

    elif posture_type == "sitting":
        lms[0] = DummyLandmark(0.0, 0.55, 0.0)         # Head
        lms[11] = DummyLandmark(-0.20, 0.35, 0.0)      # Shoulders
        lms[12] = DummyLandmark(0.20, 0.35, 0.0)
        lms[13] = DummyLandmark(-0.28, 0.15, 0.10)     # Elbows
        lms[14] = DummyLandmark(0.28, 0.15, 0.10)
        lms[15] = DummyLandmark(-0.25, 0.0, 0.30)      # Wrists on lap
        lms[16] = DummyLandmark(0.25, 0.0, 0.30)
        lms[23] = DummyLandmark(-0.12, -0.10, 0.0)     # Hips
        lms[24] = DummyLandmark(0.12, -0.10, 0.0)
        lms[25] = DummyLandmark(-0.13, -0.10, 0.40)    # Knees flexed forward
        lms[26] = DummyLandmark(0.13, -0.10, 0.40)
        lms[27] = DummyLandmark(-0.14, -0.55, 0.40)    # Feet down
        lms[28] = DummyLandmark(0.14, -0.55, 0.40)
        lms[29] = DummyLandmark(-0.14, -0.58, 0.35)
        lms[30] = DummyLandmark(0.14, -0.58, 0.35)
        lms[31] = DummyLandmark(-0.14, -0.60, 0.50)
        lms[32] = DummyLandmark(0.14, -0.60, 0.50)

    elif posture_type == "walking":
        lms[0] = DummyLandmark(0.0, 0.68, 0.0)
        lms[11] = DummyLandmark(-0.20, 0.45, 0.0)
        lms[12] = DummyLandmark(0.20, 0.45, 0.0)
        # Left arm swinging back, right arm swinging forward
        lms[13] = DummyLandmark(-0.25, 0.22, -0.15)
        lms[14] = DummyLandmark(0.25, 0.22, 0.15)
        lms[15] = DummyLandmark(-0.28, 0.02, -0.25)
        lms[16] = DummyLandmark(0.28, 0.02, 0.25)
        lms[23] = DummyLandmark(-0.12, 0.0, 0.0)
        lms[24] = DummyLandmark(0.12, 0.0, 0.0)
        # Left leg stepping forward, right leg trailing back
        lms[25] = DummyLandmark(-0.13, -0.42, 0.25)    # L Knee forward
        lms[26] = DummyLandmark(0.13, -0.42, -0.20)   # R Knee back
        lms[27] = DummyLandmark(-0.14, -0.88, 0.38)    # L Ankle forward
        lms[28] = DummyLandmark(0.14, -0.86, -0.35)   # R Ankle back
        lms[29] = DummyLandmark(-0.14, -0.92, 0.32)
        lms[30] = DummyLandmark(0.14, -0.90, -0.40)
        lms[31] = DummyLandmark(-0.14, -0.94, 0.50)
        lms[32] = DummyLandmark(0.14, -0.94, -0.22)

    elif posture_type == "bending":
        lms[0] = DummyLandmark(0.0, 0.05, 0.45)        # Head pitched forward
        lms[11] = DummyLandmark(-0.20, 0.12, 0.30)
        lms[12] = DummyLandmark(0.20, 0.12, 0.30)
        lms[13] = DummyLandmark(-0.25, -0.15, 0.35)    # Arms hanging down
        lms[14] = DummyLandmark(0.25, -0.15, 0.35)
        lms[15] = DummyLandmark(-0.22, -0.40, 0.38)
        lms[16] = DummyLandmark(0.22, -0.40, 0.38)
        lms[23] = DummyLandmark(-0.12, 0.0, 0.0)
        lms[24] = DummyLandmark(0.12, 0.0, 0.0)
        lms[25] = DummyLandmark(-0.13, -0.45, 0.0)
        lms[26] = DummyLandmark(0.13, -0.45, 0.0)
        lms[27] = DummyLandmark(-0.14, -0.90, 0.0)
        lms[28] = DummyLandmark(0.14, -0.90, 0.0)

    elif posture_type in ("lying", "fallen"):
        lms[0] = DummyLandmark(0.0, 0.10, -0.75)       # Head on floor
        lms[11] = DummyLandmark(-0.18, 0.08, -0.50)    # Shoulders
        lms[12] = DummyLandmark(0.18, 0.08, -0.50)
        lms[13] = DummyLandmark(-0.35, 0.06, -0.45)    # Arms sprawled
        lms[14] = DummyLandmark(0.35, 0.06, -0.45)
        lms[15] = DummyLandmark(-0.48, 0.05, -0.40)
        lms[16] = DummyLandmark(0.48, 0.05, -0.40)
        lms[23] = DummyLandmark(-0.12, 0.05, 0.0)      # Hips
        lms[24] = DummyLandmark(0.12, 0.05, 0.0)
        lms[25] = DummyLandmark(-0.15, 0.05, 0.40)     # Knees
        lms[26] = DummyLandmark(0.15, 0.05, 0.40)
        lms[27] = DummyLandmark(-0.15, 0.05, 0.85)     # Feet
        lms[28] = DummyLandmark(0.15, 0.05, 0.85)

    return lms


class TestVisualPostureValidation(unittest.TestCase):

    def setUp(self):
        self.vis = SkeletonVisualizer3D(width=500, height=600)

    def test_all_postures_render_correctly(self):
        postures = ["standing", "sitting", "walking", "bending", "lying", "fallen"]

        for post in postures:
            pose = create_posture(post)
            is_fall = (post == "fallen")
            act_label = "Fallen" if is_fall else post.capitalize()

            # Render multiple frames for EMA camera to settle
            for _ in range(3):
                canvas = self.vis.render(
                    world_landmarks=pose,
                    activity_label=act_label,
                    confidence=0.95,
                    is_fall=is_fall,
                )

            self.assertEqual(canvas.shape, (600, 500, 3))
            self.assertEqual(canvas.dtype, np.uint8)

            # Ensure skeleton was actually drawn (canvas has substantial rendered content)
            self.assertGreater(np.count_nonzero(canvas), 5000)

            # Check that alert red is present in fallen mode
            if is_fall:
                # PALETTE['fall'] is (50, 50, 255)
                red_pixels = np.sum((canvas[:, :, 2] > 200) & (canvas[:, :, 0] < 120))
                self.assertGreater(red_pixels, 50, "Fallen posture should feature prominent red alerts")


if __name__ == "__main__":
    unittest.main()

