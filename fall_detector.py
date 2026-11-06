"""
End-to-end fall detection pipeline.
Layer: Orchestration of Detection -> Tracking -> Pose -> Features -> Score -> State Machine -> Alerts -> Logging -> Visualization
Optimized for high-throughput CPU inference with real-time profiling and adaptive pose tracking.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from dashboard import Dashboard, draw_person_box
from feature_extractor import FeatureExtractor
from fall_logger import FallLogger
from person_tracker import PersonTracker
from pose_estimator import PoseEstimator
from temporal_state_machine import TemporalStateMachine
from weighted_scoring import WeightedScorer
from yolo_detector import YOLODetector
from alarm_system import AlarmSystem
from activity_monitor import ActivityMonitor
from visualizer_3d import SkeletonVisualizer3D
from pipeline_profiler import PipelineProfiler
from alert_dispatcher import AlertDispatcher


SEVERITY_RANK = {"NONE": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}


class FallDetector:
    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        pose_model_path: str = "pose_landmarker.task",
        output_path: Optional[str] = "output_fall_detected_weighted.mp4",
        display: bool = True,
        display_every_n_frames: int = 5,
        enable_3d: bool = True,
        enable_activity: bool = True,
        adaptive_pose: bool = True,
        imgsz: int = 480,
        enable_alerts: bool = True,
        alert_config_path: Optional[str] = "alert_config.json",
    ) -> None:
        self.detector = YOLODetector(model_path=yolo_model_path, imgsz=imgsz)
        self.pose = PoseEstimator(model_path=pose_model_path)
        self.tracker = PersonTracker()
        self.features = FeatureExtractor()
        self.scorer = WeightedScorer()
        self.state_machine = TemporalStateMachine()
        self.logger = FallLogger()
        self.dashboard = Dashboard()
        self.alarm = AlarmSystem()
        self.alert_dispatcher = AlertDispatcher(config_path=alert_config_path, enabled=enable_alerts)

        self.enable_3d = enable_3d
        self.enable_activity = enable_activity
        self.adaptive_pose = adaptive_pose
        self.activity_monitor = ActivityMonitor() if enable_activity else None
        self.visualizer_3d = SkeletonVisualizer3D() if enable_3d else None
        self._mouse_attached = False
        self.selected_person_id: Optional[int] = None

        self.output_path = output_path
        self.display = display
        self.display_every_n_frames = max(1, int(display_every_n_frames))
        self.fps_history = deque(maxlen=15)
        self.profiler = PipelineProfiler(rolling_window=30, enabled=True)
        self.prev_time = time.time()
        self.frame_idx = 0

    def _draw_placeholder(self, frame, text: str) -> None:
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_single_frame(
        self,
        frame: np.ndarray,
        now: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Processes a single BGR video frame through the complete AI pipeline:
        Detection -> Tracking -> Multi-Person Pose -> Biomechanical Features ->
        Scoring -> State Machine -> Alerts -> Dashboard & 3D Skeleton Render.

        Returns:
            out_frame: Composited (Left Camera + Right 3D Skeleton) BGR frame.
            telemetry: Real-time dictionary of current system state for APIs/Web dashboard.
        """
        if now is None:
            now = time.time()

        h_f, w_f = frame.shape[:2]

        if self.enable_3d and self.visualizer_3d:
            if self.visualizer_3d.height != h_f:
                self.visualizer_3d.height = h_f
                self.visualizer_3d.width = int(w_f * 0.70)

        self.frame_idx += 1
        dt = max(1e-4, now - self.prev_time)
        self.fps_history.append(dt)
        self.prev_time = now

        avg_dt = (sum(self.fps_history) / len(self.fps_history)) if self.fps_history else 0.033
        fps = (
            self.profiler.average_fps
            if (self.profiler.frame_count > 5 and self.profiler.average_fps > 0.0)
            else min(120.0, 1.0 / max(0.001, avg_dt))
        )

        any_fall_active = any(s.get("fall_detected", False) for s in self.tracker._states.values())
        recently_fell = any(
            s.get("fall_reset_time") is not None
            and (now - s["fall_reset_time"]) < 30.0
            for s in self.tracker._states.values()
            if not s.get("fall_detected", False)
        )

        # Operating confidence: 0.35 during normal monitoring (allows ByteTrack
        # secondary association for sitting/occluded persons), 0.25 on active fall
        conf = 0.25 if any_fall_active else (0.30 if recently_fell else 0.35)

        # ── 2. YOLO Person Detection ──
        self.profiler.start_stage("yolo_detection")
        detected_boxes = self.detector.detect(frame, conf=conf)
        self.profiler.end_stage("yolo_detection")

        # ── 3. ByteTrack Tracking ──
        self.profiler.start_stage("bytetrack_tracking")
        tracked = self.tracker.update(detected_boxes)
        self.profiler.end_stage("bytetrack_tracking")

        active_fall_count = 0
        t_pose_accum = 0.0
        t_feat_accum = 0.0
        t_score_accum = 0.0
        t_act_accum = 0.0

        poses_computed_this_frame = 0
        max_steady_poses_per_frame = 2

        for person_id, box, state in tracked:
            is_emergency = bool(state.get("fall_detected", False) or state.get("candidate_frames", 0) > 0)
            has_last_pose = state.get("last_pose") is not None

            # ── 4. MediaPipe Pose Estimation (Adaptive Multi-Person) ──
            should_estimate_pose = True
            if self.adaptive_pose and has_last_pose and not is_emergency:
                prev_b = state.get("prev_box")
                if prev_b is not None:
                    d_box = (
                        abs(box[0] - prev_b[0])
                        + abs(box[1] - prev_b[1])
                        + abs(box[2] - prev_b[2])
                        + abs(box[3] - prev_b[3])
                    )
                    if d_box < 10.0 and (state.get("pose_step", 0) % 2 == 1):
                        should_estimate_pose = False

                if should_estimate_pose and len(tracked) >= 3 and poses_computed_this_frame >= max_steady_poses_per_frame:
                    should_estimate_pose = False

            state["pose_step"] = state.get("pose_step", 0) + 1
            state["prev_box"] = list(box)

            t_start_pose = time.perf_counter()
            if should_estimate_pose:
                pose_result = self.pose.estimate(frame, box)
                if not is_emergency:
                    poses_computed_this_frame += 1
            else:
                pose_result = state.get("last_pose")
            t_pose_accum += (time.perf_counter() - t_start_pose) * 1000.0

            state["frame_h"] = frame.shape[0]

            # ── 5. Biomechanical Feature Extraction ──
            t_start_feat = time.perf_counter()
            features = self.features.extract(state, box, pose_result)
            t_feat_accum += (time.perf_counter() - t_start_feat) * 1000.0

            # ── 6. Fall Scoring ──
            t_start_score = time.perf_counter()
            score_result = self.scorer.score_features(features)
            t_score_accum += (time.perf_counter() - t_start_score) * 1000.0

            state["torso_angle"]  = features.torso_angle
            state["aspect_ratio"] = features.aspect_ratio
            state["hip_velocity"] = features.hip_velocity
            state["upward_hip_velocity"] = features.upward_hip_velocity
            state["height_drop"]  = features.height_drop
            state["visibility"]   = features.visibility
            state["fall_score"]   = score_result.score
            state["signal_scores"]  = score_result.signal_scores
            state["head_velocity"] = features.head_velocity
            state["shoulder_hip_separation"] = features.shoulder_hip_separation
            state["knee_collapse"] = max(features.knee_collapse_left, features.knee_collapse_right)
            state["bilateral_asymmetry"] = features.bilateral_asymmetry
            state["post_impact_stillness"] = features.post_impact_stillness
            state["torso_tilt_3d"] = features.torso_tilt_3d
            state["angle_delta"]   = features.angle_delta
            state["last_pose"] = pose_result

            # ── 7. Activity Classification & Monitoring ──
            t_start_act = time.perf_counter()
            if self.enable_activity and self.activity_monitor:
                step_dt = 1.0 / max(1.0, fps)
                self.activity_monitor.update(state, box, pose_result, features, dt=step_dt)
            t_act_accum += (time.perf_counter() - t_start_act) * 1000.0

            # ── 8. Score history & severity ──
            self.features.update_score_history(state, score_result.score)

            if state["fall_detected"]:
                prev = state.get("severity", "NONE")
                if SEVERITY_RANK[score_result.severity] > SEVERITY_RANK[prev]:
                    state["severity"] = score_result.severity
            else:
                state["severity"] = score_result.severity

            prev_fall_detected = state.get("fall_detected", False)

            # ── 9. State Machine ──
            transition = self.state_machine.update(state, score_result, now_t=now)

            # ── 10. Alarm + dashboard on confirmed fall ──
            if transition.confirmed_fall:
                state["fall_start_time"] = now
                self.dashboard.update_fall(score_result.severity)
                self.alarm.trigger()
                self.alert_dispatcher.dispatch_fall_alert(
                    frame=frame,
                    person_id=person_id,
                    severity=score_result.severity,
                    score=score_result.score,
                    duration=0.0,
                    activity=state.get("activity_label", "Unknown"),
                )

            # ── 11. Duration tracking + CSV logging ──
            if state["fall_detected"] and state.get("fall_start_time") is not None:
                state["fall_duration"] = time.time() - state["fall_start_time"]
                if transition.confirmed_fall:
                    self.logger.log(
                        frame,
                        person_id,
                        score_result.severity,
                        state["fall_duration"],
                        score=score_result.score,
                    )

            if state["fall_detected"] and state.get("fall_duration", 0) >= 5.0:
                self.alarm.escalate()

            # ── 12. Alarm reset when person recovers ──
            if prev_fall_detected and not state["fall_detected"]:
                self.alarm.reset()
                self.alert_dispatcher.dispatch_recovery_alert(
                    person_id=person_id,
                    duration=state.get("fall_duration", 0.0),
                )

            if state["fall_detected"]:
                active_fall_count += 1

            draw_person_box(frame, box, person_id, state)

        self.profiler.record_stage("mediapipe_pose", t_pose_accum)
        self.profiler.record_stage("feature_extraction", t_feat_accum)
        self.profiler.record_stage("fall_scoring", t_score_accum)
        self.profiler.record_stage("activity_recognition", t_act_accum)

        # Primary subject for 3D visualization & HUD summary
        focus_state = None
        focus_world_landmarks = None
        focus_act_label = None
        focus_act_conf = None
        focus_is_fall = False
        focus_person_id = None

        if tracked:
            user_chosen = None
            if self.selected_person_id is not None:
                for p in tracked:
                    if p[0] == self.selected_person_id:
                        user_chosen = p
                        break

            fall_candidates = [p for p in tracked if p[2].get("fall_detected")]
            warn_candidates = [p for p in tracked if p[2].get("candidate_frames", 0) > 0]

            if fall_candidates:
                chosen = fall_candidates[0]
            elif warn_candidates:
                chosen = warn_candidates[0]
            elif user_chosen is not None:
                chosen = user_chosen
            else:
                chosen = max(
                    tracked, key=lambda p: getattr(p[2].get("last_pose"), "visibility", 0)
                )

            focus_person_id = chosen[0]
            focus_state = chosen[2]
            focus_state["person_id"] = chosen[0]
            last_p = focus_state.get("last_pose")
            if last_p and last_p.world_landmarks:
                focus_world_landmarks = last_p.world_landmarks
                focus_act_label = focus_state.get("activity_label")
                focus_act_conf = focus_state.get("activity_confidence")
                focus_is_fall = focus_state.get("fall_detected", False)

        if active_fall_count > 0:
            self.dashboard.fall_active = True
            self.dashboard.fall_lost_frames = 0
        elif len(tracked) == 0 and self.dashboard.fall_active:
            self.dashboard.fall_lost_frames += 1
            if self.dashboard.fall_lost_frames > self.dashboard.FALL_LOST_TOLERANCE:
                self.dashboard.fall_active = False
                self.dashboard.fall_lost_frames = 0
        elif len(tracked) > 0 and active_fall_count == 0:
            self.dashboard.fall_active = False
            self.dashboard.fall_lost_frames = 0

        # ── Dashboard Rendering ──
        self.profiler.start_stage("dashboard_render")
        frame = self.dashboard.draw(
            frame,
            active_persons=len(tracked),
            fps=fps,
            summary_state=focus_state,
            profiler=self.profiler,
            tracked_persons=tracked,
        )
        self.profiler.end_stage("dashboard_render")

        # ── Render Interactive 3D Skeleton Side-Panel ──
        self.profiler.start_stage("visualizer_3d")
        if self.enable_3d and self.visualizer_3d:
            canvas_3d = self.visualizer_3d.render(
                world_landmarks=focus_world_landmarks,
                activity_label=focus_act_label,
                confidence=focus_act_conf,
                is_fall=focus_is_fall,
                person_id=focus_person_id,
            )
            out_frame = np.hstack([frame, canvas_3d])
        else:
            out_frame = frame
        self.profiler.end_stage("visualizer_3d")

        overall_status = "NORMAL"
        if self.dashboard.fall_active:
            overall_status = "FALL DETECTED"
        elif any(p[2].get("candidate_frames", 0) > 0 for p in tracked):
            overall_status = "WARNING"
        elif any(p[2].get("fall_reset_time") is not None and (now - p[2]["fall_reset_time"]) < 8.0 for p in tracked):
            overall_status = "RECOVERY"

        def _serialize_world_landmarks(lms):
            if not lms:
                return None
            return [
                [
                    round(float(lm.x), 4),
                    round(float(lm.y), 4),
                    round(float(lm.z), 4),
                    round(float(getattr(lm, "visibility", 1.0)), 2),
                ]
                for lm in lms
            ]

        telemetry = {
            "active_persons": len(tracked),
            "fall_active": bool(self.dashboard.fall_active),
            "total_falls": int(self.dashboard.total_falls),
            "last_fall_time": str(self.dashboard.last_fall_time),
            "fps": round(float(fps), 1),
            "focus_person_id": focus_person_id,
            "focus_world_landmarks": _serialize_world_landmarks(focus_world_landmarks),
            "overall_status": overall_status,
            "camera_frame": frame,
            "persons": [
                {
                    "id": p[0],
                    "box": [int(v) for v in p[1]],
                    "activity": str(p[2].get("activity_label", "Unknown")),
                    "confidence": round(float(p[2].get("activity_confidence", 0.0)) * 100, 1) if p[2].get("activity_confidence") is not None else None,
                    "fall_detected": bool(p[2].get("fall_detected", False)),
                    "severity": str(p[2].get("severity", "NONE")),
                    "fall_score": round(float(p[2].get("fall_score", 0.0)), 2),
                    "fall_duration": round(float(p[2].get("fall_duration", 0.0)), 1),
                    "world_landmarks": _serialize_world_landmarks(getattr(p[2].get("last_pose"), "world_landmarks", None)),
                }
                for p in tracked
            ],
            "profiler": {
                "avg_fps": round(float(self.profiler.average_fps), 1),
            }
        }

        return out_frame, telemetry

    def process(self, video_path: Union[str, int]) -> Optional[str]:
        # On Windows, try DirectShow for instant webcam initialization
        if isinstance(video_path, int) and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video/webcam source: {video_path}")

        # Set low buffer size on webcams to prevent frame queuing latency
        if isinstance(video_path, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps_src = (
            float(raw_fps)
            if (raw_fps is not None and math.isfinite(raw_fps) and raw_fps > 0)
            else 25.0
        )
        w_src = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_src = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.enable_3d and self.visualizer_3d and w_src > 0 and h_src > 0:
            w_3d = int(w_src * 0.70)
            self.visualizer_3d.width = w_3d
            self.visualizer_3d.height = h_src
            out_w = w_src + w_3d
        else:
            out_w = max(1, w_src)

        out = None
        if self.output_path and str(self.output_path).strip() and out_w > 1 and h_src > 1:
            out = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps_src,
                (out_w, h_src),
            )

        frame_idx = 0

        try:
            while True:
                self.profiler.start_frame()

                # ── 1. Camera Frame Acquisition ──
                self.profiler.start_stage("camera_acquire")
                ret, frame = cap.read()
                self.profiler.end_stage("camera_acquire")

                if not ret or frame is None or frame.size == 0:
                    break

                out_frame, telemetry = self.process_single_frame(frame, now=time.time())
                frame_idx += 1

                # ── Frame Output & Preview ──
                self.profiler.start_stage("frame_output")
                if out is None and self.output_path and str(self.output_path).strip():
                    out = cv2.VideoWriter(
                        self.output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps_src,
                        (out_frame.shape[1], out_frame.shape[0]),
                    )
                if out is not None:
                    out.write(out_frame)

                if self.display and frame_idx % self.display_every_n_frames == 0:
                    win_title = "AI Fall Detection & Activity Monitoring (3D Skeleton)"
                    if self.enable_3d and not self._mouse_attached:
                        cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
                        self.visualizer_3d.attach_to_window(win_title, x_offset=w_src)
                        self._mouse_attached = True

                    cv2.imshow(win_title, out_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    # Multi-person interactive focus selector:
                    # 1-9: Select Person ID directly
                    # 0 or 'a': Reset to auto emergency priority
                    # TAB (9): Cycle to next detected person
                    if ord("1") <= key <= ord("9"):
                        self.selected_person_id = key - ord("0")
                    elif key in (ord("0"), ord("a"), ord("A")):
                        self.selected_person_id = None
                    elif key == 9:  # TAB
                        active_pids = [p["id"] for p in telemetry.get("persons", [])]
                        if active_pids:
                            if self.selected_person_id in active_pids:
                                cur_idx = active_pids.index(self.selected_person_id)
                                self.selected_person_id = active_pids[(cur_idx + 1) % len(active_pids)]
                            else:
                                self.selected_person_id = active_pids[0]

                    if self.enable_3d and self.visualizer_3d:
                        self.visualizer_3d.handle_key(key)
                self.profiler.end_stage("frame_output")

                self.profiler.end_frame()
                self.profiler.end_stage("frame_output")

                self.profiler.end_frame()

        finally:
            cap.release()
            if out is not None:
                out.release()
            self.pose.close()
            self.alarm.close()
            self.alert_dispatcher.close()
            cv2.destroyAllWindows()

        print("\n" + "=" * 55)
        print("[INFO] Session ended successfully.")
        if self.output_path:
            print(f"[INFO] Output recorded to : {self.output_path}")
        print(f"[INFO] Incident log file  : fall_events.csv")
        print(f"[INFO] Total falls logged : {self.dashboard.total_falls}")
        print("=" * 55)
        if self.profiler.frame_count > 0:
            print(self.profiler.format_report())

        return self.output_path
