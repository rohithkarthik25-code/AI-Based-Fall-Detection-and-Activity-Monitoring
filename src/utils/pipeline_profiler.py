"""
High-Resolution Pipeline Profiler for AI-Based Fall Detection & Activity Monitoring.
Measures execution latency across all 10 pipeline stages using time.perf_counter().
Computes rolling averages and summary diagnostics for real-time performance tracking.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StageStats:
    name: str
    history: deque = field(default_factory=lambda: deque(maxlen=30))

    def add(self, duration_ms: float) -> None:
        self.history.append(duration_ms)

    @property
    def mean(self) -> float:
        return sum(self.history) / len(self.history) if self.history else 0.0

    @property
    def latest(self) -> float:
        return self.history[-1] if self.history else 0.0


class PipelineProfiler:
    """
    Lightweight, high-resolution performance profiler for the fall detection pipeline.
    Tracks all 10 standard processing stages with rolling averages over N frames.
    """

    STAGES = [
        "camera_acquire",
        "yolo_detection",
        "bytetrack_tracking",
        "mediapipe_pose",
        "feature_extraction",
        "fall_scoring",
        "activity_recognition",
        "visualizer_3d",
        "dashboard_render",
        "frame_output",
        "total_frame",
    ]

    def __init__(self, rolling_window: int = 30, enabled: bool = True) -> None:
        self.enabled = enabled
        self.rolling_window = rolling_window
        self.stats: Dict[str, StageStats] = {
            stage: StageStats(name=stage, history=deque(maxlen=rolling_window))
            for stage in self.STAGES
        }
        self._current_starts: Dict[str, float] = {}
        self.frame_count: int = 0
        self._frame_start_t: float = 0.0

    def start_frame(self) -> None:
        """Marks the start of an entire frame pipeline iteration."""
        if not self.enabled:
            return
        self._frame_start_t = time.perf_counter()

    def end_frame(self) -> float:
        """Marks the end of a frame iteration and records total latency."""
        if not self.enabled or self._frame_start_t == 0.0:
            return 0.0
        total_ms = (time.perf_counter() - self._frame_start_t) * 1000.0
        self.stats["total_frame"].add(total_ms)
        self.frame_count += 1
        return total_ms

    def start_stage(self, stage: str) -> None:
        """Records timestamp at the beginning of a specific pipeline stage."""
        if not self.enabled:
            return
        self._current_starts[stage] = time.perf_counter()

    def end_stage(self, stage: str) -> float:
        """Records completion of a stage and appends elapsed milliseconds."""
        if not self.enabled:
            return 0.0
        t0 = self._current_starts.pop(stage, None)
        if t0 is None:
            return 0.0
        duration_ms = (time.perf_counter() - t0) * 1000.0
        if stage in self.stats:
            self.stats[stage].add(duration_ms)
        return duration_ms

    def record_stage(self, stage: str, duration_ms: float) -> None:
        """Directly records elapsed time in milliseconds for a stage."""
        if not self.enabled:
            return
        if stage in self.stats:
            self.stats[stage].add(duration_ms)

    @property
    def current_fps(self) -> float:
        latest = self.stats["total_frame"].latest
        return (1000.0 / latest) if latest > 0.0 else 0.0

    @property
    def average_fps(self) -> float:
        mean_total = self.stats["total_frame"].mean
        return (1000.0 / mean_total) if mean_total > 0.0 else 0.0

    @property
    def avg_yolo_ms(self) -> float:
        return self.stats["yolo_detection"].mean

    @property
    def avg_mediapipe_ms(self) -> float:
        return self.stats["mediapipe_pose"].mean

    @property
    def avg_3d_render_ms(self) -> float:
        return self.stats["visualizer_3d"].mean

    @property
    def avg_total_ms(self) -> float:
        return self.stats["total_frame"].mean

    def get_summary(self) -> Dict[str, float]:
        """Returns structured summary of current performance metrics."""
        return {
            "current_fps": self.current_fps,
            "average_fps": self.average_fps,
            "avg_yolo_ms": self.avg_yolo_ms,
            "avg_bytetrack_ms": self.stats["bytetrack_tracking"].mean,
            "avg_mediapipe_ms": self.avg_mediapipe_ms,
            "avg_features_ms": self.stats["feature_extraction"].mean,
            "avg_scoring_ms": self.stats["fall_scoring"].mean,
            "avg_activity_ms": self.stats["activity_recognition"].mean,
            "avg_3d_render_ms": self.avg_3d_render_ms,
            "avg_dashboard_ms": self.stats["dashboard_render"].mean,
            "avg_frame_output_ms": self.stats["frame_output"].mean,
            "avg_total_ms": self.avg_total_ms,
        }

    def format_report(self) -> str:
        """Formats a human-readable performance profile report."""
        lines = [
            "=" * 64,
            f"PERFORMANCE PROFILER REPORT (Rolling Window: {self.rolling_window} frames)",
            "=" * 64,
            f"Current FPS               : {self.current_fps:6.2f} FPS",
            f"Average FPS               : {self.average_fps:6.2f} FPS",
            f"Total Frame Processing    : {self.avg_total_ms:6.2f} ms",
            "-" * 64,
        ]
        total_ms = max(0.001, self.avg_total_ms)
        for stage in self.STAGES:
            if stage == "total_frame":
                continue
            mean_ms = self.stats[stage].mean
            pct = (mean_ms / total_ms) * 100.0
            label = stage.replace("_", " ").title()
            lines.append(f"{label:26s}: {mean_ms:6.2f} ms ({pct:5.1f}%)")
        lines.append("=" * 64)
        return "\n".join(lines)

