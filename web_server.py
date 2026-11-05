"""
High-Performance FastAPI Web Server for Real-Time Fall Detection & 3D Skeleton Streaming.
Layer: Web Server & REST API Gateway
"""

from __future__ import annotations

import argparse
import csv
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from web_pipeline import WebVideoPipeline


# ------------------------------------------------------------------
# Request & Response Models
# ------------------------------------------------------------------

class FocusRequest(BaseModel):
    person_id: Optional[int] = None


class SourceRequest(BaseModel):
    source: str


# ------------------------------------------------------------------
# Server Setup & Lifespan
# ------------------------------------------------------------------

pipeline: Optional[WebVideoPipeline] = None
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background video ingestion pipeline
    global pipeline
    if pipeline is not None:
        pipeline.start()
    yield
    # Shutdown: Cleanly release hardware and models
    if pipeline is not None:
        pipeline.stop()


app = FastAPI(
    title="AI Fall Detection & 3D Skeleton Monitoring API",
    description="Real-time multi-person pose estimation, activity recognition, fall detection, and 3D kinematics stream.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for external frontend or mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve incident snapshots statically
if not os.path.exists("fall_logs"):
    os.makedirs("fall_logs", exist_ok=True)
app.mount("/fall_logs", StaticFiles(directory="fall_logs"), name="fall_logs")


# ------------------------------------------------------------------
# Routes & Endpoints
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Serves the main interactive monitoring dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed")
async def video_feed():
    """Streams live composited video (Camera Feed + 3D Skeleton) via MJPEG."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return StreamingResponse(
        pipeline.stream_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/telemetry")
async def get_telemetry() -> Dict[str, Any]:
    """Returns real-time telemetry: FPS, fall states, posture labels, confidence, and person counts."""
    if pipeline is None:
        return {"status": "OFFLINE"}
    return pipeline.get_telemetry()


@app.post("/api/focus")
async def set_focus(req: FocusRequest):
    """Selects which person ID to center in the interactive 3D skeleton side panel."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.set_focus_person(req.person_id)
    return {"status": "ok", "focus_person_id": req.person_id}


@app.post("/api/source")
async def change_source(req: SourceRequest):
    """Switches active video source (e.g. webcam index or video filepath)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    src = int(req.source) if req.source.isdigit() else req.source
    success = pipeline.switch_source(src)
    return {"status": "ok" if success else "error", "source": req.source}


@app.get("/api/incidents")
async def get_incidents():
    """Returns historical confirmed falls parsed from fall_events.csv."""
    csv_path = "fall_events.csv"
    if not os.path.exists(csv_path):
        return []

    incidents = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                incidents.append(row)
    except Exception as e:
        print(f"[Error reading incidents CSV]: {e}")

    return incidents


@app.post("/api/test_alarm")
async def test_alarm():
    """Dispatches a manual test alarm notification to configured channels."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    pipeline.detector.alarm.trigger()
    pipeline.detector.alert_dispatcher.dispatch_fall_alert(
        frame=None,
        person_id=99,
        severity="SEVERE",
        score=0.99,
        duration=1.2,
        activity="Manual Web Test",
    )
    return {"status": "ok", "message": "Test alert successfully triggered across notification channels!"}


# ------------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AI Fall Detection Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--source", default="0", help="Video source (webcam index or video path)")
    parser.add_argument("--no-3d", action="store_true", help="Disable 3D skeleton in stream")
    parser.add_argument("--no-activity", action="store_true", help="Disable activity recognition")
    parser.add_argument("--no-alerts", action="store_true", help="Disable Telegram/Email alerts")
    return parser.parse_args()


def create_pipeline_from_args(args) -> WebVideoPipeline:
    src = int(args.source) if str(args.source).isdigit() else args.source
    return WebVideoPipeline(
        source=src,
        enable_3d=not args.no_3d,
        enable_activity=not args.no_activity,
        enable_alerts=not args.no_alerts,
    )


if __name__ == "__main__":
    cli_args = parse_args()
    pipeline = create_pipeline_from_args(cli_args)
    print("\n" + "=" * 65)
    print(f"🚀 AI Fall Detection Web Server launching on http://localhost:{cli_args.port}")
    print(f"📹 Active video input: {cli_args.source}")
    print(f"🌐 Access locally: http://localhost:{cli_args.port}")
    print(f"📱 Access on local network: http://<your-ip>:{cli_args.port}")
    print("=" * 65 + "\n")
    uvicorn.run(app, host=cli_args.host, port=cli_args.port, log_level="info")
