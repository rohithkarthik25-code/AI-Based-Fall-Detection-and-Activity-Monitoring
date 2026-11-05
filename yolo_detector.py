"""
YOLO-based person detector with ByteTrack.
Layer: Detection + Tracking
Optimized for high-throughput CPU execution with inference_mode and batched tensor extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from ultralytics import YOLO


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def as_xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        person_class_id: int = 0,
        confidence: float = 0.35,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 480,
        device: Optional[str] = "cpu",
    ) -> None:
        self.model = YOLO(model_path)
        self.person_class_id = person_class_id
        self.confidence = confidence
        self.tracker = tracker
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame, conf: float | None = None) -> List[TrackedBox]:
        threshold = conf if conf is not None else self.confidence   
        
        with torch.inference_mode():
            results = self.model.track(
                frame,
                imgsz=self.imgsz,
                classes=[self.person_class_id],
                conf=threshold,
                iou=0.45,
                tracker=self.tracker,
                persist=True,
                verbose=False,
                device=self.device,
            )[0]

        boxes: List[TrackedBox] = []
        if results.boxes is None or results.boxes.id is None:
            return boxes

        ids = results.boxes.id.int().cpu().tolist()
        xyxy = results.boxes.xyxy.int().cpu().tolist()
        confs = results.boxes.conf.float().cpu().tolist()

        for tid, (x1, y1, x2, y2), c in zip(ids, xyxy, confs):
            boxes.append(TrackedBox(
                track_id=tid,
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=round(c, 4),
            ))

        return boxes
