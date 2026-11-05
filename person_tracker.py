"""
Per-person state store.
ID assignment is handled upstream by ByteTrack.
Layer: Tracking (state only)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Tuple

from yolo_detector import TrackedBox


class PersonTracker:
    def __init__(self, max_missing: int = 20) -> None:
        self.max_missing = max_missing
        self._states: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _new_state(self, box: TrackedBox) -> dict:
        return {
            "box": box.as_xyxy(),
            "last_seen": time.time(),
            "missing": 0,
            "score_history": deque(maxlen=10),
            "severity": "NONE",
            "fall_detected": False,
            "fall_start_time": None,
            "fall_duration": 0.0,
            "candidate_frames": 0,
            "normal_frames": 0,
            "fall_reset_time": None,
            "prev_head_y": None,
            "prev_left_hip_y": None,
            "prev_right_hip_y": None,
            "landmark_history": deque(maxlen=6),
            "motion_energy_history": deque(maxlen=10),
        }

    def _iou(self, a: list, b: list) -> float:
        """Standard Intersection over Union."""
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(areaA + areaB - inter)
    
    def _iomin(self, a: list, b: list) -> float:
        """Intersection over the SMALLER box — catches partial overlaps of the same body."""
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / float(min(areaA, areaB))

    def _nms_tracks(self, tracks: List[Tuple], iou_thresh: float = 0.65, iomin_thresh: float = 0.80) -> List[Tuple]:
        """
        Suppress duplicate bounding boxes of the SAME person while preserving
        distinct individuals standing or moving in close proximity.

        ByteTrack handles multi-person tracking and assigns unique track IDs.
        We only suppress if two boxes truly represent the same physical body:
        1. High IoU (>= 0.65): heavy geometric overlap of the same person.
        2. High IoMIN (>= 0.80) with aligned horizontal centers: sub-box nested inside
           a full body box (e.g. accidental torso-only detection).
        """
        if len(tracks) <= 1:
            return tracks

        def priority(t):
            _, box, state = t
            area = (box[2] - box[0]) * (box[3] - box[1])
            is_fall = int(state.get("fall_detected", False))
            is_cand = int(state.get("candidate_frames", 0) > 0)
            return (is_fall, is_cand, area)

        tracks = sorted(tracks, key=priority, reverse=True)

        kept = []
        for track in tracks:
            _, box, _ = track
            suppress = False
            b_cx = (box[0] + box[2]) / 2.0
            b_w = max(1.0, float(box[2] - box[0]))

            for _, kept_box, _ in kept:
                iou = self._iou(box, kept_box)
                iomin = self._iomin(box, kept_box)
                k_cx = (kept_box[0] + kept_box[2]) / 2.0
                k_w = max(1.0, float(kept_box[2] - kept_box[0]))

                # 1. Direct duplicate detection (high IoU)
                if iou >= iou_thresh:
                    suppress = True
                    break

                # 2. Nested duplicate sub-box (high containment + aligned center)
                center_dist_x = abs(b_cx - k_cx)
                min_w = min(b_w, k_w)
                if iomin >= iomin_thresh and (center_dist_x / min_w) < 0.25:
                    suppress = True
                    break

            if not suppress:
                kept.append(track)

        return kept

    def _try_inherit_state(self, tb: TrackedBox) -> dict | None:
        """
        When ByteTrack assigns a new ID to a person we were already tracking
        (e.g. after a 1-2 frame dropout), find the nearest missing track and
        inherit its state — preserving fall_detected, score_history,
        candidate_frames, severity, etc. — instead of starting from scratch.
        """
        new_box = tb.as_xyxy()
        new_cx = (new_box[0] + new_box[2]) / 2
        new_cy = (new_box[1] + new_box[3]) / 2

        best_tid = None
        best_dist = float("inf")

        for tid, state in self._states.items():
            if state["missing"] == 0:
                continue  # only consider currently-missing tracks
            box = state["box"]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            dist = ((new_cx - cx) ** 2 + (new_cy - cy) ** 2) ** 0.5
            box_diag = ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5
            if dist < box_diag * 1.5 and dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is not None:
            inherited = self._states.pop(best_tid)  # remove old ID entry
            inherited["box"] = new_box
            inherited["missing"] = 0
            inherited["last_seen"] = time.time()
            return inherited

        return None

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def update(self, tracked_boxes: List[TrackedBox]) -> List[Tuple[int, list, dict]]:
        """
        Merge ByteTrack results into the state store.

        Returns list of (track_id, xyxy, state) — same shape as before
        so downstream layers need no changes.
        """
        seen_ids = set()

        for tb in tracked_boxes:
            tid = tb.track_id
            seen_ids.add(tid)

            if tid not in self._states:
                # New ID from ByteTrack — check if it's actually a person
                # we were already tracking that just had a brief dropout
                inherited = self._try_inherit_state(tb)
                self._states[tid] = inherited if inherited else self._new_state(tb)
            else:
                self._states[tid]["box"] = tb.as_xyxy()
                self._states[tid]["missing"] = 0
                self._states[tid]["last_seen"] = time.time()

        # Age out tracks ByteTrack has dropped
        for tid in list(self._states):
            if tid not in seen_ids:
                self._states[tid]["missing"] += 1
                if self._states[tid]["missing"] >= self.max_missing:
                    del self._states[tid]

        # Active tracks: seen in this frame, OR recently fallen (up to 5 frames tolerance)
        results = [
            (tid, s["box"], s)
            for tid, s in self._states.items()
            if s["missing"] == 0 or (s.get("fall_detected", False) and s["missing"] <= 5)
        ]
        return self._nms_tracks(results)