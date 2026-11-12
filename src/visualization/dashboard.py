"""
Professional Dual-Panel Dashboard & HUD Overlay for AI-Based Fall Detection and Activity Monitoring.
Provides structured information hierarchy across Detection Status, Activity Recognition,
Person Tracking, Fall Statistics, and Performance Metrics with a modern dark theme.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Professional Color Palette (BGR)
# ─────────────────────────────────────────────────────────────────────────────
COLOR_BG_CARD       = (22, 19, 28)       # Translucent dark slate navy fill
COLOR_CARD_BORDER   = (55, 52, 68)       # Subtle card outline
COLOR_DIVIDER       = (42, 38, 52)       # Section separator line
COLOR_TEXT_TITLE    = (235, 240, 245)    # High-contrast title
COLOR_TEXT_LABEL    = (148, 158, 172)    # Muted section label
COLOR_TEXT_VALUE    = (220, 226, 235)    # Primary readout text
COLOR_TEXT_DIM      = (110, 120, 132)    # Secondary / detail text

# Status Alert Colors
STATUS_NORMAL_COLOR   = (80, 210, 60)    # Emerald Green
STATUS_WARNING_COLOR  = (0, 215, 255)    # Amber Yellow
STATUS_FALL_COLOR     = (40, 50, 240)    # Crimson Red
STATUS_RECOVERY_COLOR = (245, 160, 45)   # Cerulean Blue

STATE_COLORS = {
    "NORMAL":    STATUS_NORMAL_COLOR,
    "WARNING":   STATUS_WARNING_COLOR,
    "CANDIDATE": STATUS_WARNING_COLOR,
    "FALLEN":    STATUS_FALL_COLOR,
    "RECOVERY":  STATUS_RECOVERY_COLOR,
}

SEVERITY_COLORS = {
    "NONE":     STATUS_NORMAL_COLOR,
    "MILD":     (0, 220, 220),
    "MODERATE": (0, 140, 255),
    "SEVERE":   STATUS_FALL_COLOR,
}

# MediaPipe pose connections (landmark index pairs)
POSE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

# Key landmarks to draw dots on
KEY_LANDMARKS = {
    11: "L.Sh", 12: "R.Sh",
    13: "L.El", 14: "R.El",
    23: "L.Hip", 24: "R.Hip",
    25: "L.Kn",  26: "R.Kn",
    27: "L.Ank", 28: "R.Ank",
}


class Dashboard:
    """
    Renders structured HUD overlay on the 2D camera feed.
    Information Hierarchy:
      1. Detection Status (NORMAL, WARNING, FALL DETECTED, RECOVERY) with glowing status pill
      2. Activity Information (Activity Label, Confidence %, Subject Tracking ID)
      3. Person Tracking (Active count, IDs, Status)
      4. Fall Statistics (Total confirmed, Last fall time, Fall score, Severity)
      5. Real-Time Performance (Current FPS, Rolling Avg FPS, Pipeline latency ms)
      6. Posture Distribution (Sitting %, Standing %, Walking %)
      7. Bottom Emergency Alert Banners (Confirmed Fall, Recovery, Track Lost)
    """

    FALL_LOST_TOLERANCE = 30

    def __init__(self) -> None:
        self.total_falls: int = 0
        self.last_fall_time: str = "None"
        self.fall_active: bool = False
        self.fall_lost_frames: int = 0
        self.frame_times: deque = deque(maxlen=10)

    def update_fall(self, severity: str = "NONE") -> None:
        """Records confirmed fall incident timestamp and increments counter."""
        self.total_falls += 1
        self.last_fall_time = time.strftime("%H:%M:%S")
        self.fall_active = True

    def draw(
        self,
        frame: np.ndarray,
        active_persons: int,
        fps: float,
        summary_state: Optional[Dict[str, Any]] = None,
        profiler: Any = None,
        tracked_persons: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Draws the professional structured HUD panel on the camera feed.
        Maintains original aspect ratio and strictly respects panel boundaries.
        """
        fh, fw = frame.shape[:2]

        # ── 1. Determine System Detection Status (Hierarchy A & Step 5) ──
        is_fall = self.fall_active or bool(summary_state and summary_state.get("fall_detected", False))

        is_recovery = False
        if not is_fall and summary_state:
            reset_t = summary_state.get("fall_reset_time")
            if reset_t is not None and (time.time() - reset_t) < 8.0:
                is_recovery = True
            elif summary_state.get("is_recovering", False):
                is_recovery = True

        is_warning = False
        if not is_fall and not is_recovery and summary_state:
            cand = summary_state.get("candidate_frames", 0)
            sc = summary_state.get("fall_score", 0.0)
            if cand > 0 or sc > 0.40:
                is_warning = True

        if is_fall:
            status_text = "FALL DETECTED"
            status_color = STATUS_FALL_COLOR
        elif is_recovery:
            status_text = "RECOVERY"
            status_color = STATUS_RECOVERY_COLOR
        elif is_warning:
            status_text = "WARNING"
            status_color = STATUS_WARNING_COLOR
        else:
            status_text = "NORMAL"
            status_color = STATUS_NORMAL_COLOR

        # ── 2. Top-Left Structured HUD Card ──
        card_x = 12
        card_y = 12
        card_w = min(360, max(280, fw - 24))

        has_postures = bool(summary_state and "time_percentages" in summary_state)
        card_h = 224 if has_postures else 198
        card_h = min(card_h, fh - 24)

        if card_w > 50 and card_h > 50 and card_y + card_h <= fh and card_x + card_w <= fw:
            # Subtle translucent background using in-place blend
            roi = frame[card_y:card_y+card_h, card_x:card_x+card_w]
            if roi.size > 0:
                cv2.addWeighted(roi, 0.22, np.full_like(roi, 20), 0.78, 0, roi)
                # Outer border
                cv2.rectangle(frame, (card_x, card_y), (card_x+card_w, card_y+card_h), COLOR_CARD_BORDER, 1)
                # 2px top status accent line
                cv2.line(frame, (card_x, card_y), (card_x+card_w, card_y), status_color, 2)

            # ── Header: Title & Prominent Status Pill ──
            cv2.putText(
                frame,
                "AI SAFETY MONITOR",
                (card_x + 12, card_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                COLOR_TEXT_TITLE,
                1,
                cv2.LINE_AA,
            )

            # Prominent status pill docked on the right
            pill_text = f"● {status_text}"
            (tw, th), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            pill_x2 = card_x + card_w - 10
            pill_x1 = pill_x2 - tw - 12
            pill_y1 = card_y + 8
            pill_y2 = card_y + 26
            if pill_x1 > card_x + 150:
                cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), (28, 25, 36), -1)
                cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), status_color, 1)
                cv2.putText(
                    frame,
                    pill_text,
                    (pill_x1 + 6, pill_y1 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    status_color,
                    1,
                    cv2.LINE_AA,
                )

            # Header separator line
            cv2.line(frame, (card_x + 10, card_y + 32), (card_x + card_w - 10, card_y + 32), COLOR_DIVIDER, 1)

            # ── Row 1: Activity Information (Hierarchy B) ──
            y_r1 = card_y + 54
            cv2.putText(frame, "ACTIVITY", (card_x + 12, y_r1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_LABEL, 1, cv2.LINE_AA)
            if summary_state and summary_state.get("activity_label"):
                act_str = str(summary_state["activity_label"]).upper()
                conf_val = summary_state.get("activity_confidence")
                conf_str = f" — {int(conf_val * 100)}%" if conf_val is not None else ""
                pid_val = summary_state.get("person_id")
                pid_str = f" [P{pid_val}]" if pid_val is not None else ""
                act_display = f"{act_str}{conf_str}{pid_str}"
                
                # Activity highlight color
                act_color = STATUS_NORMAL_COLOR if act_str == "STANDING" else (
                    STATUS_WARNING_COLOR if act_str in ("WALKING", "BENDING") else (
                        STATUS_RECOVERY_COLOR if act_str == "SITTING" else (
                            STATUS_FALL_COLOR if is_fall else COLOR_TEXT_VALUE
                        )
                    )
                )
            else:
                act_display = "STANDBY (No Subject)"
                act_color = COLOR_TEXT_DIM

            cv2.putText(frame, act_display, (card_x + 95, y_r1), cv2.FONT_HERSHEY_SIMPLEX, 0.44, act_color, 1, cv2.LINE_AA)

            # ── Row 2: Person Tracking (Hierarchy C) ──
            y_r2 = card_y + 80
            cv2.putText(frame, "TRACKING", (card_x + 12, y_r2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_LABEL, 1, cv2.LINE_AA)
            if tracked_persons:
                pids = [f"P{p[0]}" for p in tracked_persons]
                track_display = f"{active_persons} Detected  [{', '.join(pids)}]"
            elif summary_state and summary_state.get("person_id") is not None:
                track_display = f"{active_persons} Detected  [P{summary_state['person_id']}]"
            else:
                track_display = f"{active_persons} Detected (Idle)" if active_persons > 0 else "0 Persons (Idle)"
            cv2.putText(frame, track_display, (card_x + 95, y_r2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_VALUE, 1, cv2.LINE_AA)

            # ── Row 3: Fall Statistics (Hierarchy D) ──
            y_r3 = card_y + 106
            cv2.putText(frame, "INCIDENTS", (card_x + 12, y_r3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_LABEL, 1, cv2.LINE_AA)
            stats_display = f"{self.total_falls} Confirmed  (Last: {self.last_fall_time})"
            cv2.putText(frame, stats_display, (card_x + 95, y_r3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_VALUE, 1, cv2.LINE_AA)

            # ── Row 4: Fall Score & Severity (Hierarchy D) ──
            y_r4 = card_y + 132
            cv2.putText(frame, "SEVERITY", (card_x + 12, y_r4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_LABEL, 1, cv2.LINE_AA)
            score = summary_state.get("fall_score", 0.0) if summary_state else 0.0
            sev = summary_state.get("severity", "NONE") if summary_state else "NONE"
            sev_col = SEVERITY_COLORS.get(sev, COLOR_TEXT_VALUE)
            score_display = f"Score: {score:.2f}  |  Level: {sev}"
            cv2.putText(frame, score_display, (card_x + 95, y_r4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sev_col, 1, cv2.LINE_AA)

            # ── Row 5: Real-Time Performance (Hierarchy E) ──
            y_r5 = card_y + 158
            cv2.putText(frame, "PERF", (card_x + 12, y_r5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_LABEL, 1, cv2.LINE_AA)
            if profiler is not None and getattr(profiler, "frame_count", 0) > 0:
                avg_fps = profiler.average_fps
                lat_ms = profiler.avg_total_ms
                perf_display = f"{fps:.1f} FPS (avg: {avg_fps:.1f}) | {lat_ms:.1f} ms"
            else:
                perf_display = f"{fps:.1f} FPS"
            cv2.putText(frame, perf_display, (card_x + 95, y_r5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, STATUS_NORMAL_COLOR, 1, cv2.LINE_AA)

            # ── Row 6: Posture Distribution (if available) ──
            if has_postures and summary_state:
                pcts = summary_state["time_percentages"]
                y_r6 = card_y + 184
                cv2.putText(frame, "POSTURES", (card_x + 12, y_r6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_LABEL, 1, cv2.LINE_AA)
                breakdown = f"Sit: {pcts.get('Sitting', 0):.0f}% | Stand: {pcts.get('Standing', 0):.0f}% | Walk: {pcts.get('Walking', 0):.0f}%"
                cv2.putText(frame, breakdown, (card_x + 95, y_r6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # ── 3. Emergency Status Indicators & Banners (Step 5) ──
        if is_fall:
            pid_str = f"P{summary_state.get('person_id', 1)}" if summary_state else "P1"
            sev_str = summary_state.get("severity", "SEVERE") if summary_state else "SEVERE"
            dur_val = summary_state.get("fall_duration", 0.0) if summary_state else 0.0
            sc_val = summary_state.get("fall_score", 0.0) if summary_state else 0.0

            banner_text = f"! CRITICAL: FALL DETECTED — {pid_str} | {sev_str} | {dur_val:.1f}s | Score: {sc_val:.2f} !"
            self._draw_emergency_banner(frame, banner_text, STATUS_FALL_COLOR)

        elif is_recovery:
            pid_str = f"P{summary_state.get('person_id', 1)}" if summary_state else "P1"
            banner_text = f"ℹ RECOVERY DETECTED — {pid_str} STABILIZING"
            self._draw_emergency_banner(frame, banner_text, STATUS_RECOVERY_COLOR)

        elif self.fall_active and active_persons == 0:
            banner_text = "! CRITICAL ALERT: PERSON DOWN ON GROUND — TRACK LOST !"
            self._draw_emergency_banner(frame, banner_text, STATUS_FALL_COLOR)

        return frame

    def _draw_emergency_banner(self, frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
        """Renders high-visibility emergency alert banner at the bottom of the camera feed."""
        fh, fw = frame.shape[:2]
        banner_w = min(580, fw - 24)
        banner_h = 36
        bx1 = (fw - banner_w) // 2
        by1 = fh - banner_h - 14
        bx2 = bx1 + banner_w
        by2 = by1 + banner_h

        if bx1 >= 0 and by1 >= 0 and bx2 <= fw and by2 <= fh:
            roi = frame[by1:by2, bx1:bx2]
            if roi.size > 0:
                cv2.addWeighted(roi, 0.20, np.full_like(roi, 16), 0.80, 0, roi)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
            tx = max(bx1 + 10, bx1 + (banner_w - tw) // 2)
            ty = by1 + (banner_h + th) // 2
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.50, color, 1, cv2.LINE_AA)


def _draw_skeleton(frame: np.ndarray, box: list, pose_result: Any) -> None:
    """Draws MediaPipe keypoints and skeleton connections onto the 2D frame."""
    if pose_result is None or not getattr(pose_result, "has_pose", False):
        return

    x1, y1, x2, y2 = box
    h, w = pose_result.crop_shape[:2]
    landmarks = pose_result.landmarks
    if not landmarks:
        return

    def to_frame(lm):
        px = int(x1 + lm.x * w)
        py = int(y1 + lm.y * h)
        return px, py

    # Draw skeleton connections
    for a, b in POSE_CONNECTIONS:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        vis_a = getattr(landmarks[a], "visibility", 0.0)
        vis_b = getattr(landmarks[b], "visibility", 0.0)
        if vis_a < 0.3 or vis_b < 0.3:
            continue
        pt_a = to_frame(landmarks[a])
        pt_b = to_frame(landmarks[b])
        cv2.line(frame, pt_a, pt_b, (0, 200, 255), 1, cv2.LINE_AA)

    # Draw landmark dots
    for idx, label in KEY_LANDMARKS.items():
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        vis = getattr(lm, "visibility", 0.0)
        if vis < 0.3:
            continue
        pt = to_frame(lm)
        # Hips in yellow, others in cyan
        dot_color = (0, 255, 255) if idx in (23, 24) else (255, 255, 0)
        cv2.circle(frame, pt, 4, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_factor_panel(frame: np.ndarray, box: list, state: dict) -> None:
    """
    Renders non-intrusive factor panel beside the bounding box showing
    key fall signals with visual bars and raw numeric values.
    """
    x1, y1, x2, y2 = box
    sig   = state.get("signal_scores", {})
    score = state.get("fall_score", 0.0)
    angle = state.get("torso_angle")
    ar    = state.get("aspect_ratio", 0.0)
    vel   = state.get("hip_velocity", 0.0)
    drop  = state.get("height_drop", 0.0)
    vis   = state.get("visibility", 0.0)

    factors = [
        ("Score",   score,                         (220, 220, 220)),
        ("Angle",   sig.get("angle",       0.0),   (100, 200, 255)),
        ("A/R",     sig.get("ratio",       0.0),   (100, 255, 200)),
        ("Drop",    sig.get("height_drop", 0.0),   (255, 200, 100)),
        ("Vel",     sig.get("velocity",    0.0),   (255, 100, 200)),
        ("HeadV",   sig.get("head_vel",    0.0),   (255, 180,  80)),
        ("Sep",     sig.get("separation",  0.0),   (180, 255, 100)),
        ("Tilt3D",  sig.get("tilt_3d",     0.0),   (100, 180, 255)),
        ("Knee",    sig.get("knee",        0.0),   (255, 120, 120)),
        ("Still",   sig.get("stillness",   0.0),   (120, 255, 180)),
        ("Persist", sig.get("persistence", 0.0),   (200, 100, 255)),
        ("Vis",     vis,                           (180, 180, 180)),
    ]

    bar_max = 68
    panel_w = bar_max + 58
    row_h   = 16
    panel_h = len(factors) * row_h + 6

    # Position to the right of box; fallback below box if no space
    panel_x = x2 + 8
    panel_y = y1
    fits_right = (panel_x + panel_w < frame.shape[1]) and (panel_y + panel_h < frame.shape[0])

    if not fits_right:
        panel_x = max(4, min(frame.shape[1] - panel_w - 4, x1))
        panel_y = y2 + 6
        if panel_y + panel_h >= frame.shape[0]:
            return  # Skip if insufficient room to prevent clipping

    px1 = max(0, panel_x - 2)
    py1 = max(0, panel_y - 2)
    px2 = min(frame.shape[1], panel_x + panel_w)
    py2 = min(frame.shape[0], panel_y + panel_h)

    roi = frame[py1:py2, px1:px2]
    if roi.size > 0:
        cv2.addWeighted(roi, 0.25, np.full_like(roi, 20), 0.75, 0, roi)
        cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_CARD_BORDER, 1)

    for i, (label, val, color) in enumerate(factors):
        ry = panel_y + 4 + i * row_h
        cv2.putText(frame, f"{label}:", (panel_x + 4, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR_TEXT_LABEL, 1, cv2.LINE_AA)

        bx = panel_x + 46
        cv2.rectangle(frame, (bx, ry + 2), (bx + bar_max, ry + 12), (40, 38, 48), -1)

        fill  = max(0.0, min(1.0, float(val)))
        bar_w = int(fill * bar_max)
        if bar_w > 0:
            r = int(255 * fill)
            g = int(255 * (1 - fill))
            cv2.rectangle(frame, (bx, ry + 2), (bx + bar_w, ry + 12), (0, g, r), -1)

        cv2.putText(frame, f"{val:.2f}", (bx + bar_max + 3, ry + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

    # Raw metrics row
    raw_y = panel_y + panel_h + 2
    if angle is not None and raw_y + 12 < frame.shape[0]:
        cv2.putText(
            frame,
            f"ang={angle:.0f}°  ar={ar:.2f}  vel={vel:.2f}  drop={drop:.2f}",
            (panel_x, raw_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )


def draw_person_box(frame: np.ndarray, box: list, pid: int, state: dict) -> None:
    """
    Renders person bounding box with docked status tag, 2D MediaPipe skeleton,
    and non-intrusive factor panel. Avoids excessive overlay on the person's body.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = box

    # Safe coordinate clamping
    x1_c = max(0, min(fw - 1, int(x1)))
    y1_c = max(0, min(fh - 1, int(y1)))
    x2_c = max(0, min(fw - 1, int(x2)))
    y2_c = max(0, min(fh - 1, int(y2)))

    # Determine state & color
    is_fallen = bool(state.get("fall_detected", False))
    is_cand   = state.get("candidate_frames", 0) > 0
    is_rec    = bool(state.get("fall_reset_time") is not None and (time.time() - state["fall_reset_time"]) < 8.0)

    if is_fallen:
        color = STATUS_FALL_COLOR
        status_tag = "FALLEN"
    elif is_rec:
        color = STATUS_RECOVERY_COLOR
        status_tag = "RECOVERY"
    elif is_cand:
        color = STATUS_WARNING_COLOR
        status_tag = "WARNING"
    else:
        color = STATUS_NORMAL_COLOR
        status_tag = "NORMAL"

    # Draw bounding box with corner accents
    if x2_c > x1_c and y2_c > y1_c:
        cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), color, 2)
        corner_len = min(16, (x2_c - x1_c) // 3, (y2_c - y1_c) // 3)
        if corner_len > 4:
            cv2.line(frame, (x1_c, y1_c), (x1_c + corner_len, y1_c), color, 3)
            cv2.line(frame, (x1_c, y1_c), (x1_c, y1_c + corner_len), color, 3)
            cv2.line(frame, (x2_c, y1_c), (x2_c - corner_len, y1_c), color, 3)
            cv2.line(frame, (x2_c, y1_c), (x2_c, y1_c + corner_len), color, 3)
            cv2.line(frame, (x1_c, y2_c), (x1_c + corner_len, y2_c), color, 3)
            cv2.line(frame, (x1_c, y2_c), (x1_c, y2_c - corner_len), color, 3)
            cv2.line(frame, (x2_c, y2_c), (x2_c - corner_len, y2_c), color, 3)
            cv2.line(frame, (x2_c, y2_c), (x2_c, y2_c - corner_len), color, 3)

    # Docked Tag Header above Bounding Box
    act_label = state.get("activity_label")
    conf      = state.get("activity_confidence")
    sev       = state.get("severity", "NONE")
    dur       = state.get("fall_duration", 0.0)
    score     = state.get("fall_score", 0.0)

    if is_fallen:
        tag_str = f"P{pid} | FALLEN [{sev}] {dur:.1f}s"
    elif is_rec:
        tag_str = f"P{pid} | RECOVERY"
    elif is_cand:
        tag_str = f"P{pid} | WARNING (Score: {score:.2f})"
    elif act_label:
        conf_str = f" — {int(conf * 100)}%" if conf is not None else ""
        tag_str = f"P{pid} | {act_label.upper()}{conf_str}"
    else:
        tag_str = f"P{pid} | {status_tag}"

    (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    tag_w = tw + 14
    tag_h = th + 10

    # Place tag above box if room, otherwise dock inside top of box
    tag_y2 = y1_c if y1_c >= tag_h else min(fh, y1_c + tag_h)
    tag_y1 = max(0, tag_y2 - tag_h)
    tag_x1 = max(0, min(fw - tag_w, x1_c))
    tag_x2 = tag_x1 + tag_w

    if tag_x2 <= fw and tag_y2 <= fh and tag_w > 0 and tag_h > 0:
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), (24, 20, 32), -1)
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1)
        cv2.putText(
            frame,
            tag_str,
            (tag_x1 + 6, tag_y1 + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 245, 250),
            1,
            cv2.LINE_AA,
        )

    # 2D skeleton overlay on person
    _draw_skeleton(frame, box, state.get("last_pose"))

    # Key biomechanical factor panel
    _draw_factor_panel(frame, box, state)
