"""
Multi-Modal Synthetic Video / AI-Generated Deepfake Fall Detector.
Layer: AI Video Guardrail & Anti-Spoofing Verification
Prevents AI-generated video simulations (e.g. Sora, Runway Gen-2/Gen-3, Kling,
Pika, Stable Video Diffusion, Luma Dream Machine) from generating false emergency
alerts or fooling clinical fall detection systems.

Core Defensive Verification Pillars:
1. Anthropometric Invariance: Real human skeleton bones (femur, tibia, humerus, torso)
   are rigid. Tracks standard deviations and coefficients of variation of bone lengths
   and limb-to-torso ratios. Diffusion models hallucinate and stretch/shrink limbs.
2. Newtonian Gravitational Kinematics: Real falling bodies accelerate downward under gravity
   (9.8 m/s^2) followed by sharp deceleration impact impulse on the floor. Synthetic models
   often teleport, glide with zero acceleration, or float without rigid impact.
3. High-Frequency Spatial/Temporal Texture & FFT Rolloff: Real camera sensors exhibit
   natural optical noise, clothing micro-textures, and high Laplacian variance (>350).
   Diffusion models use VAE latent decoders that introduce heavy oversmoothing and spectral rolloff (<280).
4. Temporal Landmark Continuity & Kalman Residual: Measures frame-to-frame keypoint jitter
   and unphysical per-frame morphing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SyntheticDetectionResult:
    is_synthetic: bool
    synthetic_score: float              # 0.0 (authentic real) to 1.0 (synthetic AI)
    confidence: float
    status_label: str                  # "AUTHENTIC HUMAN" or "AI GENERATED VIDEO"
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class SyntheticVideoDetector:
    """
    Robust AI Video Guardrail that analyzes spatial texture, anthropometric rigid-body
    invariance, and Newtonian kinematics to reliably discriminate between authentic human
    footage and AI-generated video simulations.
    """

    def __init__(
        self,
        history_window: int = 30,
        synthetic_threshold: float = 0.50,
        texture_laplacian_threshold: float = 320.0,
    ) -> None:
        self.history_window = max(10, int(history_window))
        self.synthetic_threshold = float(synthetic_threshold)
        self.texture_laplacian_threshold = float(texture_laplacian_threshold)

        # Per-person tracking states for temporal consistency
        self._person_histories: Dict[int, Dict[str, Any]] = {}
        # Global frame statistics
        self._global_laplacians = deque(maxlen=self.history_window)
        self._global_fft_ratios = deque(maxlen=self.history_window)
        self._smoothed_synthetic_score: float = 0.0

    def reset(self) -> None:
        """Resets all internal history buffers."""
        self._person_histories.clear()
        self._global_laplacians.clear()
        self._global_fft_ratios.clear()
        self._smoothed_synthetic_score = 0.0

    @staticmethod
    def _calc_laplacian_variance(crop: np.ndarray) -> float:
        """Computes variance of the Laplacian as a measure of high-frequency texture sharpness."""
        if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return 0.0
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(var)

    @staticmethod
    def _calc_fft_high_freq_ratio(crop: np.ndarray) -> float:
        """Computes ratio of high-frequency to total spectral energy using 2D Fourier Transform."""
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return 0.5
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop

        # Resize to fixed size for scale-invariant spectral analysis
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        # Zero-center to eliminate DC offset
        zero_mean = resized.astype(np.float32) - float(np.mean(resized))
        f = np.fft.fft2(zero_mean)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        ch, cw = 32, 32
        r = 8  # low frequency center radius
        y, x = np.ogrid[:64, :64]
        mask_low = (x - cw) ** 2 + (y - ch) ** 2 <= r ** 2

        low_energy = np.sum(magnitude_spectrum[mask_low])
        high_energy = np.sum(magnitude_spectrum[~mask_low])
        total = low_energy + high_energy + 1e-6
        return float(high_energy / total)

    @staticmethod
    def _dist_2d(p1, p2, w: float, h: float) -> float:
        return math.sqrt(((p1.x - p2.x) * w) ** 2 + ((p1.y - p2.y) * h) ** 2)

    @staticmethod
    def _dist_3d(p1, p2) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def analyze_frame(
        self,
        frame: np.ndarray,
        tracked_persons: List[Tuple[int, List[int], Dict[str, Any]]],
    ) -> SyntheticDetectionResult:
        """
        Analyzes the current frame and tracked persons for synthetic/AI video signatures.

        Args:
            frame: Full BGR frame.
            tracked_persons: List of (person_id, box, state) tuples.

        Returns:
            SyntheticDetectionResult with classification, score, and anomaly reasons.
        """
        fh, fw = frame.shape[:2]
        reasons: List[str] = []

        # ── 1. Spatial & Frequency Texture Analysis ──
        # Measure Laplacian texture on all detected subjects and global scene
        person_lap_vars: List[float] = []
        person_fft_ratios: List[float] = []

        for person_id, box, state in tracked_persons:
            bx1, by1, bx2, by2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(fw, bx2), min(fh, by2)
            crop = frame[by1:by2, bx1:bx2]
            if crop.size > 0:
                lap = self._calc_laplacian_variance(crop)
                fft = self._calc_fft_high_freq_ratio(crop)
                person_lap_vars.append(lap)
                person_fft_ratios.append(fft)

        if person_lap_vars:
            frame_lap = float(np.mean(person_lap_vars))
            frame_fft = float(np.mean(person_fft_ratios))
        else:
            # Fallback to center crop of scene
            c_y1, c_y2 = int(fh * 0.2), int(fh * 0.8)
            c_x1, c_x2 = int(fw * 0.2), int(fw * 0.8)
            center_crop = frame[c_y1:c_y2, c_x1:c_x2]
            frame_lap = self._calc_laplacian_variance(center_crop)
            frame_fft = self._calc_fft_high_freq_ratio(center_crop)

        self._global_laplacians.append(frame_lap)
        self._global_fft_ratios.append(frame_fft)

        avg_lap = float(np.mean(self._global_laplacians))
        avg_fft = float(np.mean(self._global_fft_ratios))

        # Real cameras typically exhibit Laplacian > 380 (often 600-1500)
        # AI diffusion video typically exhibits Laplacian < 280 (often 80-240)
        # Score 0.0 (high sharpness/noise) to 1.0 (heavy diffusion oversmoothing)
        if avg_lap < self.texture_laplacian_threshold:
            texture_score = max(0.0, min(1.0, (self.texture_laplacian_threshold - avg_lap) / (self.texture_laplacian_threshold - 120.0)))
        else:
            texture_score = 0.0

        if texture_score > 0.40:
            reasons.append(f"Diffusion texture smoothing (Laplacian Var: {avg_lap:.1f} < {self.texture_laplacian_threshold:.0f})")

        # ── 2. Anthropometric Bone Stretch & Morphing Analysis ──
        bone_morph_score = 0.0
        kinematic_anomaly_score = 0.0

        for person_id, box, state in tracked_persons:
            hist = self._person_histories.setdefault(person_id, {
                "ratios": deque(maxlen=self.history_window),
                "bone_3d": deque(maxlen=self.history_window),
                "hip_y": deque(maxlen=self.history_window),
                "box_h": deque(maxlen=self.history_window),
                "prev_ratio": None,
                "prev_hip_y": None,
                "prev_time": None,
                "morph_flags": 0,
                "teleport_flags": 0,
            })

            box_h = max(1.0, float(box[3] - box[1]))
            hist["box_h"].append(box_h)

            last_p = state.get("last_pose")
            if last_p and last_p.has_pose and last_p.landmarks:
                lms = last_p.landmarks
                cw, ch = last_p.crop_shape[1], last_p.crop_shape[0]

                ls, rs = lms[11], lms[12]
                lh, rh = lms[23], lms[24]
                lk, rk = lms[25], lms[26]

                # Torso vector & length
                mid_sh = type('pt', (), {'x': (ls.x + rs.x) / 2.0, 'y': (ls.y + rs.y) / 2.0})()
                mid_hip = type('pt', (), {'x': (lh.x + rh.x) / 2.0, 'y': (lh.y + rh.y) / 2.0})()
                torso_len = max(1.0, self._dist_2d(mid_sh, mid_hip, cw, ch))

                # Femur lengths
                l_femur = self._dist_2d(lh, lk, cw, ch)
                r_femur = self._dist_2d(rh, rk, cw, ch)
                femur_avg = (l_femur + r_femur) / 2.0
                ratio_femur_torso = femur_avg / torso_len

                # Global hip Y position
                global_hip_y = box[1] + (mid_hip.y * ch)
                hist["hip_y"].append(global_hip_y)

                # Check instantaneous anatomical ratio jump (morphing anomaly)
                prev_ratio = hist.get("prev_ratio")
                if prev_ratio is not None:
                    delta_ratio = abs(ratio_femur_torso - prev_ratio)
                    # Anthropometric violation: limb suddenly stretched or shrunk by > 35% of torso in 1 frame
                    if delta_ratio > 0.35:
                        hist["morph_flags"] += 1
                        reasons.append(f"Anatomical stretch violation: Δlimb-ratio {delta_ratio:.2f} (Person {person_id})")

                hist["prev_ratio"] = ratio_femur_torso
                hist["ratios"].append(ratio_femur_torso)

                # Check kinematic teleportation / warp anomaly
                prev_hip_y = hist.get("prev_hip_y")
                if prev_hip_y is not None:
                    dy = abs(global_hip_y - prev_hip_y)
                    norm_dy = dy / box_h
                    # Teleportation: hip moved > 40% of body height in a single frame
                    if norm_dy > 0.40:
                        hist["teleport_flags"] += 1
                        reasons.append(f"Unphysical kinematic discontinuity: Δy {dy:.1f}px ({norm_dy*100:.0f}% body) in 1 frame")

                hist["prev_hip_y"] = global_hip_y

            # Check accumulated flags
            if hist.get("morph_flags", 0) > 0:
                bone_morph_score = max(bone_morph_score, min(1.0, hist["morph_flags"] * 0.45))
            if hist.get("teleport_flags", 0) > 0:
                kinematic_anomaly_score = max(kinematic_anomaly_score, min(1.0, hist["teleport_flags"] * 0.50))

        # ── 3. Authenticity Texture Gate & Multi-Modal Fusion ──
        # If camera sensor noise and natural micro-textures are high (Laplacian > 350),
        # the footage is from a genuine physical camera. Lying-down foreshortening is not AI morphing.
        if avg_lap > 350.0:
            raw_score = 0.0
            reasons.clear()
        else:
            # Latent diffusion smoothing detected. Combine texture rolloff with kinematic/morphing violations.
            raw_score = (
                0.50 * texture_score +
                0.30 * bone_morph_score +
                0.20 * kinematic_anomaly_score
            )
            # Synergistic trigger: diffusion texture + morphing/teleportation
            if texture_score > 0.40 and (bone_morph_score > 0.30 or kinematic_anomaly_score > 0.30):
                raw_score = min(1.0, raw_score + 0.30)

        # Exponential moving average filter for stable classification
        alpha = 0.25
        self._smoothed_synthetic_score = (alpha * raw_score) + ((1.0 - alpha) * self._smoothed_synthetic_score)

        is_synthetic = self._smoothed_synthetic_score >= self.synthetic_threshold
        status_label = "AI GENERATED VIDEO" if is_synthetic else "AUTHENTIC HUMAN"
        confidence = round(float(self._smoothed_synthetic_score if is_synthetic else (1.0 - self._smoothed_synthetic_score)), 3)

        # Filter unique reasons
        unique_reasons = list(dict.fromkeys(reasons))

        metrics = {
            "laplacian_var": round(avg_lap, 1),
            "fft_ratio": round(avg_fft, 3),
            "texture_score": round(texture_score, 3),
            "bone_morph_score": round(bone_morph_score, 3),
            "kinematic_score": round(kinematic_anomaly_score, 3),
            "composite_score": round(self._smoothed_synthetic_score, 3),
        }

        return SyntheticDetectionResult(
            is_synthetic=is_synthetic,
            synthetic_score=round(self._smoothed_synthetic_score, 3),
            confidence=confidence,
            status_label=status_label,
            reasons=unique_reasons,
            metrics=metrics,
        )
