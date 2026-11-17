"""
High-Performance FastAPI Web Server for Real-Time Fall Detection & 3D Skeleton Streaming.
Layer: Web Server & REST API Gateway
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

# Add src and subpackages to sys.path
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
for _p in [_ROOT, _SRC, _SRC / "core", _SRC / "web", _SRC / "alerts", _SRC / "visualization", _SRC / "utils"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.alerts.email_alert_service import (
    claim_and_send_email_alert,
    AuthPermissionError,
    SSRFValidationError,
)

from web_pipeline import WebVideoPipeline


# ------------------------------------------------------------------
# Request & Response Models
# ------------------------------------------------------------------

class EmailAlertRequest(BaseModel):
    event_id: str

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

if os.path.exists("Demo Videos"):
    app.mount("/Demo Videos", StaticFiles(directory="Demo Videos"), name="demo_videos_space")
    app.mount("/demo_videos", StaticFiles(directory="Demo Videos"), name="demo_videos")

if os.path.exists("frontend"):
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


# ------------------------------------------------------------------
# Routes & Endpoints
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
@app.get("/home.html", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
async def home_page(request: Request):
    """Serves the public marketing landing & hero page (Home Portal)."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard.html", response_class=HTMLResponse)
@app.get("/monitor", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serves the main interactive monitoring dashboard."""
    return templates.TemplateResponse(request=request, name="dashboard.html")


@app.get("/config.js")
async def config_js():
    """Serves the central marketing & branding configuration script."""
    candidates = [
        Path("frontend/config.js"),
        Path("templates/config.js"),
        _ROOT / "frontend" / "config.js",
        _ROOT / "templates" / "config.js",
        _ROOT.parent.parent / "frontend" / "config.js",
        _ROOT.parent.parent / "templates" / "config.js",
    ]
    for candidate in candidates:
        if candidate.exists():
            return FileResponse(str(candidate.resolve()), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="config.js not found")


@app.get("/video_feed")
async def video_feed(mode: str = "camera"):
    """
    Streams live video via MJPEG.
    mode="camera"   -> Clean live camera feed with 2D AI bounding boxes (paired with Three.js 3D viewport)
    mode="combined" -> Classic side-by-side feed (Camera + OpenCV Python 3D visualizer)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return StreamingResponse(
        pipeline.stream_mjpeg(mode=mode),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Real-time zero-delay WebSocket stream of 3D metric landmarks & posture telemetry for Three.js."""
    await websocket.accept()
    last_sent_seq = -1
    try:
        while True:
            if pipeline is not None:
                telemetry = pipeline.get_telemetry()
                cur_seq = telemetry.get("frame_seq", 0)
                if cur_seq != last_sent_seq:
                    await websocket.send_json(telemetry)
                    last_sent_seq = cur_seq
            await asyncio.sleep(0.005)  # 5ms fast poll check -> immediate push upon frame completion
    except (WebSocketDisconnect, Exception):
        pass


@app.get("/api/telemetry")
async def get_telemetry() -> Dict[str, Any]:
    """Returns real-time telemetry: FPS, fall states, posture labels, confidence, and 3D world landmarks."""
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


@app.post("/api/alerts/send-email")
async def send_email_alert_endpoint(req: EmailAlertRequest, request: Request):
    """
    Receives { event_id } from the authenticated frontend to trigger an emergency email alert.
    Validates JWT server-side via Supabase Auth, executes atomic DB claim, verifies ownership,
    fetches snapshot safely from Supabase Storage API, and dispatches email via SMTP.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        result = claim_and_send_email_alert(
            event_id=req.event_id,
            auth_token=auth_header
        )
        return result
    except AuthPermissionError as e:
        status_code = 403 if "Forbidden" in str(e) else 401
        raise HTTPException(status_code=status_code, detail=str(e))
    except SSRFValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[send_email_alert_endpoint] Server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal email service error: {e}")


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
