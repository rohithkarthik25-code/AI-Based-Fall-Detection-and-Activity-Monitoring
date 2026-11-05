"""
Thread-safe background video ingestion and AI inference pipeline for web streaming.
Layer: Web Video Ingestion & Pipeline Orchestration
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from fall_detector import FallDetector


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
        jpeg_quality: int = 82,
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
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        self._current_jpeg: Optional[bytes] = None
        self._current_telemetry: Dict[str, Any] = {
            "status": "INITIALIZING",
            "active_persons": 0,
            "fall_active": False,
            "total_falls": 0,
            "last_fall_time": "--:--:--",
            "fps": 0.0,
            "focus_person_id": None,
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

        self._running = True
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()
        print(f"[WebVideoPipeline] Pipeline thread started for source: {self.source}")

    def stop(self) -> None:
        """Gracefully halts background ingestion and releases capture resources."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        with self._lock:
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
            if self._cap is not None:
                self._cap.release()
            self.source = new_source
            self._cap = self._open_capture(new_source)
            success = bool(self._cap and self._cap.isOpened())
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

            try:
                out_frame, telemetry = self.detector.process_single_frame(frame, now=time.time())
                telemetry["source"] = str(self.source)

                # JPEG compression
                ret_enc, jpeg_buf = cv2.imencode(".jpg", out_frame, encode_params)
                if ret_enc:
                    jpeg_bytes = jpeg_buf.tobytes()
                    with self._lock:
                        self._current_jpeg = jpeg_bytes
                        self._current_telemetry = telemetry

            except Exception as e:
                print(f"[WebVideoPipeline] Inference error: {e}")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Web Streaming & Telemetry Endpoints
    # ------------------------------------------------------------------

    def get_telemetry(self) -> Dict[str, Any]:
        """Thread-safe snapshot of latest pipeline telemetry."""
        with self._lock:
            return dict(self._current_telemetry)

    def stream_mjpeg(self):
        """Generator yielding MJPEG multipart frames for HTTP streaming response."""
        while self._running:
            with self._lock:
                frame_bytes = self._current_jpeg

            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            # ~40-50 FPS stream cadence
            time.sleep(0.022)
