"""
Unit tests for the Synthetic Video / AI-Generated Deepfake Detector.
Layer: Anti-Spoofing & AI Video Guardrail Verification
"""

import unittest
from types import SimpleNamespace
import numpy as np

from synthetic_video_detector import SyntheticVideoDetector, SyntheticDetectionResult


class TestSyntheticVideoDetector(unittest.TestCase):
    def setUp(self):
        self.detector = SyntheticVideoDetector(
            history_window=15,
            synthetic_threshold=0.50,
            texture_laplacian_threshold=320.0,
        )

    def test_init_and_reset(self):
        self.assertEqual(self.detector.history_window, 15)
        self.assertEqual(self.detector.synthetic_threshold, 0.50)
        self.detector.reset()
        self.assertEqual(len(self.detector._global_laplacians), 0)

    def test_laplacian_variance_calculation(self):
        # Uniform blank image -> 0 variance
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        var_blank = self.detector._calc_laplacian_variance(blank)
        self.assertEqual(var_blank, 0.0)

        # High texture / random noise -> high variance
        np.random.seed(42)
        noisy = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        var_noisy = self.detector._calc_laplacian_variance(noisy)
        self.assertGreater(var_noisy, 500.0)

    def test_fft_high_freq_ratio(self):
        # Constant image has minimal high frequency ratio
        constant_img = np.full((64, 64), 128, dtype=np.uint8)
        ratio_const = self.detector._calc_fft_high_freq_ratio(constant_img)
        self.assertLessEqual(ratio_const, 0.01)

        # High frequency checkered pattern
        checkered = np.indices((64, 64)).sum(axis=0) % 2 * 255
        checkered = checkered.astype(np.uint8)
        ratio_check = self.detector._calc_fft_high_freq_ratio(checkered)
        self.assertGreater(ratio_check, 0.60)

    def test_authentic_camera_footage_scoring(self):
        # Simulate high-texture natural camera frame (Laplacian > 500)
        np.random.seed(123)
        frame = np.random.randint(40, 220, (480, 640, 3), dtype=np.uint8)
        # Person box
        box = [100, 50, 250, 400]
        
        # Consistent human pose landmarks (torso and femur lengths rigid)
        def create_consistent_pose(step):
            lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(33)]
            lms[11] = SimpleNamespace(x=0.4, y=0.2, z=0.0)  # Left shoulder
            lms[12] = SimpleNamespace(x=0.6, y=0.2, z=0.0)  # Right shoulder
            lms[23] = SimpleNamespace(x=0.45, y=0.5, z=0.0) # Left hip
            lms[24] = SimpleNamespace(x=0.55, y=0.5, z=0.0) # Right hip
            lms[25] = SimpleNamespace(x=0.45, y=0.7, z=0.0) # Left knee
            lms[26] = SimpleNamespace(x=0.55, y=0.7, z=0.0) # Right knee
            return SimpleNamespace(has_pose=True, landmarks=lms, crop_shape=(350, 150, 3))

        tracked = [(1, box, {"last_pose": create_consistent_pose(0)})]
        res = self.detector.analyze_frame(frame, tracked)

        self.assertFalse(res.is_synthetic)
        self.assertEqual(res.status_label, "AUTHENTIC HUMAN")
        self.assertLess(res.synthetic_score, 0.35)

    def test_synthetic_morphing_and_smoothing_detection(self):
        # Simulate oversmoothed AI diffusion frame (Laplacian Var < 150)
        smoothed_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Add slight soft gradient
        for y in range(480):
            smoothed_frame[y, :] = int(120 + 20 * np.sin(y / 40.0))

        box = [100, 50, 250, 400]

        def create_pose_with_limb_length(femur_y):
            lms = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(33)]
            lms[11] = SimpleNamespace(x=0.4, y=0.2, z=0.0)
            lms[12] = SimpleNamespace(x=0.6, y=0.2, z=0.0)
            lms[23] = SimpleNamespace(x=0.45, y=0.5, z=0.0)
            lms[24] = SimpleNamespace(x=0.55, y=0.5, z=0.0)
            lms[25] = SimpleNamespace(x=0.45, y=femur_y, z=0.0)
            lms[26] = SimpleNamespace(x=0.55, y=femur_y, z=0.0)
            return SimpleNamespace(has_pose=True, landmarks=lms, crop_shape=(350, 150, 3))

        # Feed frame 1 with normal limb length
        tracked1 = [(1, box, {"last_pose": create_pose_with_limb_length(0.70)})]
        _ = self.detector.analyze_frame(smoothed_frame, tracked1)

        # Feed frame 2 with sudden extreme limb collapse (ratio jump > 0.40)
        tracked2 = [(1, box, {"last_pose": create_pose_with_limb_length(0.52)})]
        res2 = self.detector.analyze_frame(smoothed_frame, tracked2)

        # Feed frame 3 with teleportation anomaly
        box_jump = [100, 250, 250, 600] # 200px jump in 1 frame
        tracked3 = [(1, box_jump, {"last_pose": create_pose_with_limb_length(0.95)})]
        res3 = self.detector.analyze_frame(smoothed_frame, tracked3)

        self.assertGreater(res3.synthetic_score, 0.45)
        self.assertTrue(any("Anatomical" in r or "Diffusion" in r or "discontinuity" in r for r in res3.reasons))


if __name__ == "__main__":
    unittest.main()
