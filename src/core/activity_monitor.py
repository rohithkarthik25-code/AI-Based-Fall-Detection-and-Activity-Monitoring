"""
Activity Monitoring & Posture Classification Module
Layer: Activity Recognition & Biomechanical Analysis

Classifies human posture and continuous activities of daily living (ADL):
- STANDING
- SITTING
- WALKING
- LYING_DOWN
- BENDING
- FALLEN

Tracks continuous session metrics including duration per activity, transition counts,
and sedentary/active ratio for healthcare and elderly monitoring.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ActivityType(str, Enum):
    STANDING = "Standing"
    SITTING = "Sitting"
    WALKING = "Walking"
    LYING_DOWN = "Lying Down"
    BENDING = "Bending"
    FALLEN = "Fallen"
    UNKNOWN = "Unknown"


# MediaPipe landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


@dataclass
class ActivityMetrics:
    """Snapshot of activity analysis for a single tracked person."""
    activity: str
    confidence: float
    torso_angle: float
    avg_knee_angle: float
    horizontal_speed: float
    current_activity_duration: float
    total_tracked_time: float
    activity_durations: Dict[str, float] = field(default_factory=dict)
    time_percentages: Dict[str, float] = field(default_factory=dict)
    posture_transitions: int = 0
    is_sedentary: bool = False


class ActivityMonitor:
    """
    Real-time posture and activity classification engine.
    Utilizes 2D/3D joint angles, bounding box geometry, and velocity vectors.
    """

    def __init__(
        self,
        debounce_frames: int = 6,
        walk_speed_threshold: float = 0.035,
        default_fps: float = 25.0,
    ) -> None:
        self.debounce_frames = debounce_frames
        self.walk_speed_threshold = walk_speed_threshold
        self.default_dt = 1.0 / max(1.0, default_fps)

    # ─────────────────────────────────────────────────────────────
    # Joint Angle Mathematics (3D with 2D fallback)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_3d_joint_angle(p_a: Any, p_b: Any, p_c: Any) -> float:
        """
        Calculate angle at joint B formed by vectors B->A and B->C in 3D space.
        Returns angle in degrees (0 to 180).
        """
        v1 = (p_a.x - p_b.x, p_a.y - p_b.y, getattr(p_a, "z", 0.0) - getattr(p_b, "z", 0.0))
        v2 = (p_c.x - p_b.x, p_c.y - p_b.y, getattr(p_c, "z", 0.0) - getattr(p_c, "z", 0.0))

        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0

        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_theta))

    @staticmethod
    def _compute_2d_joint_angle(p_a: Any, p_b: Any, p_c: Any, h: float, w: float) -> float:
        """Fallback joint angle in pixel space."""
        ax, ay = p_a.x * w, p_a.y * h
        bx, by = p_b.x * w, p_b.y * h
        cx, cy = p_c.x * w, p_c.y * h

        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)

        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 180.0

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_theta))

    def _extract_knee_angles(self, pose_result) -> Tuple[float, float, float]:
        """
        Extract (left_knee_angle, right_knee_angle, representative_knee_angle).
        Prefers metric 3D world landmarks for camera-angle invariance.
        Handles partial occlusion by weighting the more visible leg.
        """
        if not pose_result.has_pose:
            return 180.0, 180.0, 180.0

        # Check visibility from 2D landmarks if available
        l_vis = 1.0
        r_vis = 1.0
        if pose_result.landmarks and len(pose_result.landmarks) > 28:
            l_vis = getattr(pose_result.landmarks[L_KNEE], "visibility", 1.0)
            r_vis = getattr(pose_result.landmarks[R_KNEE], "visibility", 1.0)

        # Try 3D world landmarks first
        if pose_result.world_landmarks and len(pose_result.world_landmarks) > 28:
            wl = pose_result.world_landmarks
            left_knee = self._compute_3d_joint_angle(wl[L_HIP], wl[L_KNEE], wl[L_ANKLE])
            right_knee = self._compute_3d_joint_angle(wl[R_HIP], wl[R_KNEE], wl[R_ANKLE])
        elif pose_result.landmarks and len(pose_result.landmarks) > 28:
            lms = pose_result.landmarks
            h, w = pose_result.crop_shape[:2]
            left_knee = self._compute_2d_joint_angle(lms[L_HIP], lms[L_KNEE], lms[L_ANKLE], h, w)
            right_knee = self._compute_2d_joint_angle(lms[R_HIP], lms[R_KNEE], lms[R_ANKLE], h, w)
        else:
            return 180.0, 180.0, 180.0

        # If one leg is heavily occluded (profile/side view), use the visible leg
        if l_vis < 0.35 and r_vis >= 0.5:
            rep_knee = right_knee
        elif r_vis < 0.35 and l_vis >= 0.5:
            rep_knee = left_knee
        else:
            rep_knee = (left_knee + right_knee) / 2.0

        return left_knee, right_knee, rep_knee

    # ─────────────────────────────────────────────────────────────
    # Horizontal Motion & Kinematics
    # ─────────────────────────────────────────────────────────────

    def _calculate_horizontal_speed(
        self, state: Dict[str, Any], box: Tuple[int, int, int, int], dt: float
    ) -> float:
        """
        Estimates horizontal displacement normalized by box height per second.
        Used to reliably distinguish stationary standing from walking.
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        bh = max(1.0, float(y2 - y1))

        prev_cx = state.get("activity_prev_cx")
        state["activity_prev_cx"] = cx

        if prev_cx is None or dt <= 0:
            return 0.0

        dx = abs(cx - prev_cx)
        norm_speed = min(2.0, (dx / bh) / max(0.001, dt))

        # Smooth speed over recent frames
        speed_hist = state.setdefault("activity_speed_hist", deque(maxlen=6))
        speed_hist.append(norm_speed)
        return float(sum(speed_hist) / len(speed_hist))

    # ─────────────────────────────────────────────────────────────
    # Core Posture Classification Heuristics
    # ─────────────────────────────────────────────────────────────

    def _classify_raw_posture(
        self,
        state: Dict[str, Any],
        features: Any,
        avg_knee: float,
        horiz_speed: float,
    ) -> Tuple[ActivityType, float]:
        """
        Rule-based biomechanical expert classifier.
        Returns (ActivityType, confidence).
        """
        # 1. Fall detection override
        if state.get("fall_detected", False):
            return ActivityType.FALLEN, 0.99

        torso_angle = getattr(features, "torso_angle", 0.0)
        torso_tilt_3d = getattr(features, "torso_tilt_3d", torso_angle)
        effective_angle = max(torso_angle, torso_tilt_3d * 0.9)
        aspect_ratio = getattr(features, "aspect_ratio", 0.5)
        height_drop = getattr(features, "height_drop", 0.0)
        stillness = getattr(features, "post_impact_stillness", 0.0)

        # 2. Lying Down: Torso nearly horizontal, wide aspect ratio
        if effective_angle >= 58.0 and aspect_ratio >= 0.85:
            conf = min(0.98, 0.60 + (effective_angle / 90.0) * 0.3)
            return ActivityType.LYING_DOWN, conf

        # 3. Bending: Torso tilted forward significantly, but legs relatively straight
        if effective_angle >= 36.0 and avg_knee >= 135.0 and aspect_ratio < 0.95:
            conf = min(0.95, 0.55 + (effective_angle / 90.0) * 0.35)
            return ActivityType.BENDING, conf

        # 4. Sitting: Torso upright/moderate, knees bent (flexed between 65° and 130°)
        # Or significant height drop from standing baseline while upright
        is_knee_bent = (avg_knee < 135.0)
        is_torso_upright = (effective_angle < 40.0)
        is_lowered = (height_drop > 0.18 or aspect_ratio > 0.52)

        if is_knee_bent and is_torso_upright:
            conf = 0.85
            if is_lowered:
                conf += 0.10
            return ActivityType.SITTING, min(0.95, conf)

        if is_torso_upright and is_lowered and avg_knee < 148.0:
            return ActivityType.SITTING, 0.80

        # 5. Upright: Distinguish Walking vs Standing
        if is_torso_upright and avg_knee >= 135.0:
            if horiz_speed >= self.walk_speed_threshold:
                conf = min(0.95, 0.60 + (horiz_speed / (self.walk_speed_threshold * 3)) * 0.35)
                return ActivityType.WALKING, conf
            else:
                conf = 0.88
                return ActivityType.STANDING, conf

        # Default fallback
        if aspect_ratio > 1.1:
            return ActivityType.LYING_DOWN, 0.65
        return ActivityType.STANDING, 0.50

    # ─────────────────────────────────────────────────────────────
    # Temporal Smoothing & State Accumulator
    # ─────────────────────────────────────────────────────────────

    def _smooth_activity(self, state: Dict[str, Any], raw_activity: ActivityType) -> ActivityType:
        """
        Temporal window majority filter to eliminate single-frame flickering.
        Fallen state transitions bypass smoothing for immediate life safety.
        """
        if raw_activity == ActivityType.FALLEN:
            state["activity_window"] = deque([raw_activity.value], maxlen=self.debounce_frames)
            return raw_activity

        window = state.setdefault("activity_window", deque(maxlen=self.debounce_frames))
        window.append(raw_activity.value)

        # Most frequent activity in the window
        counts: Dict[str, int] = {}
        for act in window:
            counts[act] = counts.get(act, 0) + 1

        majority_act = max(counts, key=counts.get)
        return ActivityType(majority_act)

    def _update_duration_ledger(
        self, state: Dict[str, Any], smoothed_act: ActivityType, dt: float
    ) -> None:
        """Accumulates duration and tracks posture transition events."""
        ledger = state.setdefault(
            "activity_durations",
            {
                ActivityType.STANDING.value: 0.0,
                ActivityType.SITTING.value: 0.0,
                ActivityType.WALKING.value: 0.0,
                ActivityType.LYING_DOWN.value: 0.0,
                ActivityType.BENDING.value: 0.0,
                ActivityType.FALLEN.value: 0.0,
            },
        )

        state.setdefault("total_tracked_time", 0.0)
        state.setdefault("posture_transitions", 0)
        state.setdefault("current_act_duration", 0.0)

        prev_act = state.get("current_activity")
        if prev_act != smoothed_act.value:
            if prev_act is not None:
                state["posture_transitions"] += 1
            state["current_activity"] = smoothed_act.value
            state["current_act_duration"] = 0.0

        state["current_act_duration"] += dt
        state["total_tracked_time"] += dt

        act_key = smoothed_act.value
        ledger[act_key] = ledger.get(act_key, 0.0) + dt

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def update(
        self,
        state: Dict[str, Any],
        box: Tuple[int, int, int, int],
        pose_result: Any,
        features: Any,
        dt: Optional[float] = None,
    ) -> ActivityMetrics:
        """
        Main pipeline hook called per person per frame.
        """
        step_dt = dt if (dt is not None and dt > 0) else self.default_dt

        # 1. Compute biomechanics
        _, _, avg_knee = self._extract_knee_angles(pose_result)
        horiz_speed = self._calculate_horizontal_speed(state, box, step_dt)

        # 2. Raw classification
        raw_activity, confidence = self._classify_raw_posture(
            state, features, avg_knee, horiz_speed
        )

        # 3. Debounce & smooth
        smoothed_activity = self._smooth_activity(state, raw_activity)

        # 4. Duration ledger update
        self._update_duration_ledger(state, smoothed_activity, step_dt)

        # 5. Compute summary metrics
        total_time = max(0.001, state["total_tracked_time"])
        ledger = state["activity_durations"]
        percentages = {k: round((v / total_time) * 100.0, 1) for k, v in ledger.items()}

        sedentary_time = (
            ledger.get(ActivityType.SITTING.value, 0.0)
            + ledger.get(ActivityType.LYING_DOWN.value, 0.0)
        )
        is_sedentary = (total_time >= 10.0) and ((sedentary_time / total_time) > 0.65)

        # Cache on state for quick HUD access
        state["activity_label"] = smoothed_activity.value
        state["activity_confidence"] = confidence
        state["avg_knee_angle"] = avg_knee
        state["time_percentages"] = percentages
        state["is_sedentary"] = is_sedentary

        torso_angle = getattr(features, "torso_angle", 0.0)

        return ActivityMetrics(
            activity=smoothed_activity.value,
            confidence=round(confidence, 2),
            torso_angle=round(torso_angle, 1),
            avg_knee_angle=round(avg_knee, 1),
            horizontal_speed=round(horiz_speed, 3),
            current_activity_duration=round(state["current_act_duration"], 1),
            total_tracked_time=round(state["total_tracked_time"], 1),
            activity_durations={k: round(v, 1) for k, v in ledger.items()},
            time_percentages=percentages,
            posture_transitions=state["posture_transitions"],
            is_sedentary=is_sedentary,
        )

