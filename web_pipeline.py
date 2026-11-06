"""
Thread-safe background video ingestion and AI inference pipeline for web streaming.
Layer: Web Video Ingestion & Pipeline Orchestration
Optimized for zero camera buffer latency and real-time Three.js telemetry.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from fall_detector import FallDetector


class ZeroLatencyWebcamReader:
    """
    Dedicated high-speed thread that continuously empties the hardware frame buffer,
    ensuring downstream inference always receives the freshest available frame (0ms latency).
    """

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.6)


class WebVideoPipeline:
    """
    Asynchronous, non-blocking video ingestion worker.
    Continuously acquires frames from a webcam or video file, runs the FallDetector
    pipeline headlessly, caches encoded JPEG frames, and updates live telemetry.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        yolo_model_path: str = "yolov8n.pt",
        pose_model_path: str = "pose_landmarker.task",
        alert_config_path: str = "alert_config.json",
        enable_3d: bool = True,
        enable_activity: bool = True,
        enable_alerts: bool = True,
        jpeg_quality: int = 80,
    ) -> None:
        self.source = source
        self.jpeg_quality = int(jpeg_quality)

        # Initialize core AI pipeline headlessly (display=False)
        self.detector = FallDetector(
            yolo_model_path=yolo_model_path,
            pose_model_path=pose_model_path,
            output_path=None,  # No raw video file output by default
            display=False,     # Headless execution
            enable_3d=enable_3d,
            enable_activity=enable_activity,
            enable_alerts=enable_alerts,
            alert_config_path=alert_config_path,
        )

        self._cap: Optional[cv2.VideoCapture] = None
        self._webcam_reader: Optional[ZeroLatencyWebcamReader] = None
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._current_camera_jpeg: Optional[bytes] = None
        self._current_combined_jpeg: Optional[bytes] = None
        self._current_telemetry: Dict[str, Any] = {
            "status": "INITIALIZING",
            "frame_seq": 0,
            "active_persons": 0,
            "fall_active": False,
            "total_falls": 0,
            "last_fall_time": "--:--:--",
            "fps": 0.0,
            "focus_person_id": None,
            "focus_world_landmarks": None,
            "persons": [],
            "source": str(self.source),
        }

    # ------------------------------------------------------------------
    # Capture lifecycle
    # ------------------------------------------------------------------

    def _open_capture(self, source: Union[int, str]) -> cv2.VideoCapture:
        """Opens video capture with DirectShow optimization on Windows webcams."""
        if isinstance(source, int) and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def start(self) -> None:
        """Starts the background acquisition and inference thread."""
        if self._running:
            return

        self._cap = self._open_capture(self.source)
        if not self._cap or not self._cap.isOpened():
            print(f"[WebVideoPipeline] Warning: Could not open source {self.source}. Trying fallback demo video or index 0.")
            if os.path.exists("fall01.mp4"):
                self.source = "fall01.mp4"
                self._cap = self._open_capture(self.source)

        if isinstance(self.source, int) and self._cap and self._cap.isOpened():
            self._webcam_reader = ZeroLatencyWebcamReader(self._cap)

        self._running = True
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()
        print(f"[WebVideoPipeline] Pipeline thread started for source: {self.source}")

    def stop(self) -> None:
        """Gracefully halts background ingestion and releases capture resources."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.5)
        with self._lock:
            if self._webcam_reader is not None:
                self._webcam_reader.stop()
                self._webcam_reader = None
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        self.detector.pose.close()
        self.detector.alarm.close()
        self.detector.alert_dispatcher.close()
        print("[WebVideoPipeline] Pipeline stopped cleanly.")

    def switch_source(self, new_source: Union[int, str]) -> bool:
        """Dynamically switches active video capture stream."""
        with self._lock:
            if self._webcam_reader is not None:
                self._webcam_reader.stop()
                self._webcam_reader = None
            if self._cap is not None:
                self._cap.release()
            self.source = new_source
            self._cap = self._open_capture(new_source)
            success = bool(self._cap and self._cap.isOpened())
            if success and isinstance(self.source, int):
                self._webcam_reader = ZeroLatencyWebcamReader(self._cap)

            if not success:
                print(f"[WebVideoPipeline] Failed to switch to source: {new_source}")
            else:
                print(f"[WebVideoPipeline] Switched to source: {new_source}")
            return success

    def set_focus_person(self, person_id: Optional[int]) -> None:
        """Sets the active person ID for 3D skeleton visualization."""
        with self._lock:
            self.detector.selected_person_id = person_id

    # ------------------------------------------------------------------
    # Background Processing Loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        while self._running:
            # 1. Fetch freshest available frame (0ms latency reader for webcams)
            if self._webcam_reader is not None:
                ret, frame = self._webcam_reader.read()
                if not ret or frame is None:
                    time.sleep(0.006)
                    continue
            else:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self._cap.read()
                if not ret or frame is None or frame.size == 0:
                    # Video file reached EOF — loop playback automatically
                    if isinstance(self.source, str) and os.path.exists(self.source):
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                    else:
                        time.sleep(0.05)
                        continue
                time.sleep(0.02)  # Maintain standard video file rate ~25-30 FPS

            # 2. Run Headless Inference
            try:
                out_frame, telemetry = self.detector.process_single_frame(frame, now=time.time())
                telemetry["source"] = str(self.source)

                camera_frame = telemetry.pop("camera_frame", frame)

                # JPEG compression: always encode camera_frame (primary for Three.js)
                ret_cam, cam_buf = cv2.imencode(".jpg", camera_frame, encode_params)
                cam_bytes = cam_buf.tobytes() if ret_cam else None

                # Only encode side-by-side combined if enable_3d is true and different
                comb_bytes = None
                if self.detector.enable_3d and out_frame is not None and out_frame.shape != camera_frame.shape:
                    ret_comb, comb_buf = cv2.imencode(".jpg", out_frame, encode_params)
                    if ret_comb:
                        comb_bytes = comb_buf.tobytes()

                with self._lock:
                    self._current_camera_jpeg = cam_bytes
                    self._current_combined_jpeg = comb_bytes or cam_bytes
                    self._current_telemetry = telemetry

            except Exception as e:
                print(f"[WebVideoPipeline] Inference error: {e}")
                time.sleep(0.02)

    # ------------------------------------------------------------------
    # Web Streaming & Telemetry Endpoints
    # ------------------------------------------------------------------

    def get_telemetry(self) -> Dict[str, Any]:
        """Thread-safe snapshot of latest pipeline telemetry."""
        with self._lock:
            return dict(self._current_telemetry)

    def stream_mjpeg(self, mode: str = "camera"):
        """Generator yielding MJPEG multipart frames for HTTP streaming response."""
        while self._running:
            with self._lock:
                frame_bytes = self._current_camera_jpeg if mode == "camera" else self._current_combined_jpeg
                if frame_bytes is None:
                    frame_bytes = self._current_combined_jpeg or self._current_camera_jpeg

            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            # ~40-50 FPS stream cadence
            time.sleep(0.020)
