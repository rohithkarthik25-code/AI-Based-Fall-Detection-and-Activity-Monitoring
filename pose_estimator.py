"""
MediaPipe Pose Estimator.
Layer: Pose Estimation
Optimized for low-latency crop processing and robust boundary clamping.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions


@dataclass
class PoseAnalysis:
    has_pose: bool
    landmarks: Optional[list]
    world_landmarks: Optional[list]
    visibility: float
    crop_shape: tuple


class PoseEstimator:
    def __init__(self, model_path: str = "pose_landmarker.task") -> None:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame, box: List[int] | Tuple[int, int, int, int]) -> PoseAnalysis:
        x1, y1, x2, y2 = box

        # Boundary clamping within frame dimensions
        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 1, x1))
        x2 = max(0, min(fw, x2))
        y1 = max(0, min(fh - 1, y1))
        y2 = max(0, min(fh, y2))

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Reject degenerate crops early without running neural net
        if crop_w < 16 or crop_h < 24:
            return PoseAnalysis(False, None, None, 0.0, (0, 0, 0))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return PoseAnalysis(False, None, None, 0.0, (0, 0, 0))

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return PoseAnalysis(False, None, None, 0.0, crop.shape)

        landmarks = result.pose_landmarks[0]
        n_lm = len(landmarks)
        if n_lm == 0:
            return PoseAnalysis(False, None, None, 0.0, crop.shape)

        vis_total = 0.0
        for lm in landmarks:
            vis_total += getattr(lm, "visibility", 0.0)
        visibility = vis_total / n_lm

        world_landmarks = (
            result.pose_world_landmarks[0]
            if result.pose_world_landmarks
            else None
        )
        return PoseAnalysis(True, landmarks, world_landmarks, visibility, crop.shape)

    def close(self) -> None:
        self.landmarker.close()
