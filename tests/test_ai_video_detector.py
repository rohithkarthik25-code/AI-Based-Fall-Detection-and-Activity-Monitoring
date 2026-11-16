import unittest
import numpy as np

from src.core.ai_video_detector import (
    AIVideoDetector,
    VideoPreprocessor,
    FrameAnalyzer,
    TemporalAnalyzer,
    AudioAnalyzer,
    MultiModelScorer,
    FrameAnalysisResult,
    AudioAnalysisResult,
    TemporalAnalysisResult,
    AIDetectionResult,
)


class TestAIVideoDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AIVideoDetector()

    def test_preprocessor_shape_and_buffer(self):
        preprocessor = VideoPreprocessor(history_len=10, fft_size=256)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

        gray1, patch1, flow1 = preprocessor.process(frame1)
        self.assertEqual(gray1.shape, (480, 640))
        self.assertEqual(patch1.shape, (256, 256))
        self.assertIsNone(flow1)

        gray2, patch2, flow2 = preprocessor.process(frame2)
        self.assertEqual(gray2.shape, (480, 640))
        self.assertIsNotNone(flow2)
        self.assertEqual(len(preprocessor.flow_history), 1)

    def test_multi_model_scorer_real_vs_ai(self):
        scorer = MultiModelScorer()

        # Synthetic AI profile: high checkerboard ratio, low flow variance, weak/sliding accel
        ai_frame_res = FrameAnalysisResult(
            texture_checkerboard_ratio=65.0,
            spectral_slope=-3.8,
            laplacian_variance=250.0,
            sensor_noise_level=4.8,
            face_detected=False,
            face_stability=0.0,
            bone_lengths=None,
            bone_morph_score=0.45,
            frame_ai_score=0.85,
            flags=["UNet checkerboard artifact"],
        )
        ai_audio_res = AudioAnalysisResult(
            has_audio=False,
            voice_detected=False,
            audio_features_score=0.5,
            lip_sync_score=0.5,
            audio_ai_score=0.5,
            status="SILENT_NEUTRAL",
        )
        ai_temporal_res = TemporalAnalysisResult(
            motion_flow_variance=0.08,
            motion_flow_mean=0.25,
            max_downward_velocity=0.05,
            max_downward_accel=1.1,
            kinematic_gravity_score=0.85,
            temporal_jitter=0.01,
            temporal_ai_score=0.80,
            flags=["Synthetic motion smoothing"],
        )

        ai_eval = scorer.evaluate(ai_frame_res, ai_audio_res, ai_temporal_res)
        self.assertEqual(ai_eval.final_result, "AI/FAKE")
        self.assertTrue(ai_eval.is_ai_fake())
        self.assertFalse(ai_eval.is_real())
        self.assertGreaterEqual(ai_eval.ai_probability, 0.75)

        # Real video profile: low checkerboard ratio, natural camera flow variance, gravity acceleration
        real_frame_res = FrameAnalysisResult(
            texture_checkerboard_ratio=4.0,
            spectral_slope=-3.1,
            laplacian_variance=320.0,
            sensor_noise_level=5.2,
            face_detected=True,
            face_stability=0.95,
            bone_lengths={"torso": 0.3, "femur": 0.4, "tibia": 0.4, "arm": 0.3},
            bone_morph_score=0.0,
            frame_ai_score=0.05,
        )
        real_temporal_res = TemporalAnalysisResult(
            motion_flow_variance=0.45,
            motion_flow_mean=0.35,
            max_downward_velocity=0.60,
            max_downward_accel=8.5,
            kinematic_gravity_score=0.10,
            temporal_jitter=0.08,
            temporal_ai_score=0.05,
        )

        real_eval = scorer.evaluate(real_frame_res, ai_audio_res, real_temporal_res)
        self.assertEqual(real_eval.final_result, "REAL")
        self.assertTrue(real_eval.is_real())
        self.assertFalse(real_eval.is_ai_fake())
        self.assertLessEqual(real_eval.ai_probability, 0.35)

    def test_detector_process_frame(self):
        # Empty synthetic black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.detector.process_frame(frame, now=0.0, fps=25.0)
        self.assertIsInstance(res, AIDetectionResult)
        self.assertIn(res.final_result, ["REAL", "AI/FAKE", "INCONCLUSIVE"])

    def test_persistent_latch(self):
        # Once multiple AI fake frames trigger, latch remains AI/FAKE
        self.detector.confirmed_ai_fake_frames = 5
        self.detector.persistent_ai_fake = True
        consensus = self.detector.get_aggregate_verdict()
        self.assertEqual(consensus.final_result, "AI/FAKE")
        self.assertTrue(consensus.is_ai_fake())


if __name__ == "__main__":
    unittest.main()
