"""
Comprehensive Performance Benchmark Across 5 Configurations:
1. Detection Only
2. Detection + Pose Estimation
3. Full Fall Detection (No 3D, No Activity)
4. Full Fall Detection + Activity Monitoring (No 3D)
5. Full End-to-End Pipeline (2D Dashboard + Activity + Interactive 3D Skeleton)
"""

import time
import os
import cv2
import numpy as np
from typing import Dict, Any

from fall_detector import FallDetector
from pipeline_profiler import PipelineProfiler


def prepare_benchmark_video(path: str = "__bench_feed.mp4", num_frames: int = 60) -> str:
    raw_img = cv2.imread("venv/Lib/site-packages/ultralytics/assets/zidane.jpg")
    if raw_img is None:
        raise FileNotFoundError("Could not find sample image for benchmark")
    img = cv2.resize(raw_img, (640, 480))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for _ in range(num_frames):
        vw.write(img)
    vw.release()
    return path


def run_configuration(
    name: str,
    video_path: str,
    enable_3d: bool,
    enable_activity: bool,
    adaptive_pose: bool,
    imgsz: int = 480,
    pose_only: bool = False,
    detection_only: bool = False,
) -> Dict[str, Any]:
    out_video = f"__bench_{name}.mp4"
    detector = FallDetector(
        output_path=out_video,
        display=False,
        enable_3d=enable_3d,
        enable_activity=enable_activity,
        adaptive_pose=adaptive_pose,
        imgsz=imgsz,
    )

    # If detection_only, we can inspect detector directly or let it process
    cap = cv2.VideoCapture(video_path)
    profiler = PipelineProfiler(rolling_window=60)

    frame_idx = 0
    try:
        while True:
            profiler.start_frame()
            profiler.start_stage("camera_acquire")
            ret, frame = cap.read()
            profiler.end_stage("camera_acquire")
            if not ret:
                break

            frame_idx += 1
            profiler.start_stage("yolo_detection")
            boxes = detector.detector.detect(frame)
            profiler.end_stage("yolo_detection")

            if detection_only:
                profiler.end_frame()
                continue

            profiler.start_stage("bytetrack_tracking")
            tracked = detector.tracker.update(boxes)
            profiler.end_stage("bytetrack_tracking")

            t_pose = 0.0
            t_feat = 0.0
            t_score = 0.0
            t_act = 0.0

            for pid, box, state in tracked:
                t0 = time.perf_counter()
                pose_res = detector.pose.estimate(frame, box)
                t_pose += (time.perf_counter() - t0) * 1000.0

                if pose_only:
                    continue

                t0 = time.perf_counter()
                feat = detector.features.extract(state, box, pose_res)
                t_feat += (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                score = detector.scorer.score_features(feat)
                t_score += (time.perf_counter() - t0) * 1000.0

                if enable_activity and detector.activity_monitor:
                    t0 = time.perf_counter()
                    detector.activity_monitor.update(state, box, pose_res, feat, dt=0.033)
                    t_act += (time.perf_counter() - t0) * 1000.0

            profiler.record_stage("mediapipe_pose", t_pose)
            profiler.record_stage("feature_extraction", t_feat)
            profiler.record_stage("fall_scoring", t_score)
            profiler.record_stage("activity_recognition", t_act)

            if pose_only:
                profiler.end_frame()
                continue

            # Dashboard
            profiler.start_stage("dashboard_render")
            frame = detector.dashboard.draw(frame, len(tracked), 30.0)
            profiler.end_stage("dashboard_render")

            # 3D
            profiler.start_stage("visualizer_3d")
            if enable_3d and detector.visualizer_3d and tracked:
                last_p = tracked[0][2].get("last_pose")
                lms = last_p.world_landmarks if last_p else None
                canvas_3d = detector.visualizer_3d.render(lms, "Standing", 0.9, False)
            profiler.end_stage("visualizer_3d")

            profiler.end_frame()

    finally:
        cap.release()
        detector.pose.close()
        if os.path.exists(out_video):
            os.remove(out_video)

    return {
        "name": name,
        "avg_fps": profiler.average_fps,
        "total_ms": profiler.avg_total_ms,
        "yolo_ms": profiler.avg_yolo_ms,
        "track_ms": profiler.stats["bytetrack_tracking"].mean,
        "pose_ms": profiler.avg_mediapipe_ms,
        "feat_ms": profiler.stats["feature_extraction"].mean,
        "score_ms": profiler.stats["fall_scoring"].mean,
        "act_ms": profiler.stats["activity_recognition"].mean,
        "dash_ms": profiler.stats["dashboard_render"].mean,
        "viz_3d_ms": profiler.avg_3d_render_ms,
    }


def main():
    feed = "__bench_feed.mp4"
    prepare_benchmark_video(feed, num_frames=60)

    try:
        print("\n" + "=" * 80)
        print("RUNNING BENCHMARKS ACROSS 5 SYSTEM CONFIGURATIONS (CPU)")
        print("=" * 80)

        configs = [
            ("1. Detection Only", True, False, False, False, False, 480),
            ("2. Detection + Pose Estimation", False, True, False, False, False, 480),
            ("3. Full Fall Detection", False, False, False, False, False, 480),
            ("4. Full Fall Det + Activity Mon", False, False, True, False, False, 480),
            ("5. Full Pipeline (2D+3D+Act)", False, False, True, True, True, 480),
        ]

        results = []
        for label, det_only, pose_only, act, d3, adapt, imgsz in configs:
            print(f"Benchmarking: {label}...")
            res = run_configuration(
                name=label,
                video_path=feed,
                enable_3d=d3,
                enable_activity=act,
                adaptive_pose=adapt,
                imgsz=imgsz,
                pose_only=pose_only,
                detection_only=det_only,
            )
            results.append(res)

        print("\n" + "=" * 88)
        print(f"{'Configuration':<34} | {'Total Latency':<14} | {'FPS':<10} | {'YOLO (ms)':<10} | {'Pose (ms)':<10}")
        print("-" * 88)
        for r in results:
            print(
                f"{r['name']:<34} | "
                f"{r['total_ms']:7.2f} ms     | "
                f"{r['avg_fps']:6.2f} FPS | "
                f"{r['yolo_ms']:7.2f} ms | "
                f"{r['pose_ms']:7.2f} ms"
            )
        print("=" * 88)

    finally:
        if os.path.exists(feed):
            os.remove(feed)


if __name__ == "__main__":
    main()

