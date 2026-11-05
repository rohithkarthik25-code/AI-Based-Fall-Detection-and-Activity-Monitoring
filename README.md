# 🧍‍♂️➡️🛌 AI-Based Fall Detection and Activity Monitoring Using Pose Estimation with Interactive 3D Skeleton Visualization

**YOLOv8 + MediaPipe Pose Estimation with Biomechanical Feature Scoring, Real-Time Activity Recognition, Severity Classification, Smart Alerting & Interactive 3D Kinematic Visualization**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Landmarker-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-yellow)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A comprehensive real-time, CPU-only computer vision system engineered for elderly safety, independent-living monitoring, and healthcare ambient intelligence. It ingests an RGB camera stream, tracks multiple subjects in frame, extracts 33 metric 3D skeleton keypoints, continuously classifies daily postures and activities, scores 20 biomechanical signals per person to confirm falls through a temporal state machine, triggers smart two-phase audio alerts with automated incident logging, and renders a **dual-panel live feed with an interactive 3D skeleton viewport**.

Built as a Diploma Final Year Project at **Ghani Khan Choudhury Institute of Engineering & Technology** by Nabyendu Adhikary, Sagar Halder, and Prasenjit Mandal, under the guidance of Dr. Ajit Kumar Singh Yadav.

<p align="center">
  <img src="assets/demo_strip.png" alt="Fall detection and activity monitoring demo" width="100%">
</p>

---

## ✨ Key Capabilities

### 1. 🦿 AI-Based Real-Time Fall Detection
- **Multi-Person Tracking:** YOLOv8 Nano + ByteTrack with per-person persistent state and track-inheritance to survive detection dropouts.
- **14-Signal Biomechanical Scoring:** Torso angle, hip/head velocity, vertical height drop, aspect ratio, knee collapse, shoulder–hip separation, bilateral asymmetry, post-impact stillness, and 3D torso tilt combined into a continuous fall-probability score.
- **Temporal State Machine:** 8-frame confirmation window + dual-path reset (frame-count & geometry-based) to suppress transient false positives from bending, sitting down, or occlusion.
- **3-Level Severity Classification:** Monotonically non-decreasing severity grading (`MILD`, `MODERATE`, `SEVERE`).
- **Two-Phase Audio Alarm:** Soft double-beep on initial fall confirmation, escalating to a continuous high-pitch alarm if the subject remains down after 5 seconds ("long lie" prevention).
- **Automated Incident Logging:** Every confirmed fall writes timestamped entries to CSV alongside a saved screenshot.

### 2. 🚶 Activity Monitoring & Health Analytics
- **Continuous Posture Classification:** Real-time identification of Activities of Daily Living (ADL):
  - **`STANDING`** (upright torso, extended knees, minimal horizontal motion)
  - **`SITTING`** (upright torso, flexed knees between 70°–130°, lowered hip height)
  - **`WALKING`** (upright posture, cycling limb displacement, continuous horizontal motion)
  - **`BENDING`** (forward-tilted torso > 36°, legs straight, elevated hips)
  - **`LYING DOWN`** (horizontal torso > 58°, wide aspect ratio > 0.85, non-fall resting)
  - **`FALLEN`** (immediate life-safety priority override)
- **Duration Tracking & Analytics:** Accumulates exact session duration per activity, transition frequency (e.g. sit-to-stand transitions), and flags sedentary risk if sitting/lying exceeds 65% of the monitoring session.
- **HUD Integration:** Real-time activity badges on each bounding box and time percentage breakdown on the master dashboard.

### 3. 🌐 Interactive 3D Skeleton Visualization
- **Real Metric 3D Space:** Consumes MediaPipe's 33 `world_landmarks` (metric 3D coordinates in meters, hip-centered, invariant to camera tilt).
- **Live Side-by-Side Dual-Panel View:** Displays the camera feed alongside a dedicated real-time 3D skeleton rendering panel at 25–30+ FPS.
- **Full Mouse Orbit Controls:**
  - **Left-Click + Drag:** Smooth 3D camera orbit (Azimuth $-180^\circ$ to $+180^\circ$, Elevation $-85^\circ$ to $+85^\circ$).
  - **Right-Click + Drag / Mouse Wheel:** Smooth zoom in and out ($0.3\times$ to $3.5\times$).
  - **Double-Click:** Instantly resets the camera to the default perspective.
- **Keyboard View Presets:**
  - Press **`f`** $\to$ Front view
  - Press **`s`** $\to$ Side profile view
  - Press **`t`** $\to$ Top-down view
  - Press **`i`** / **`r`** $\to$ Reset to Isometric perspective
  - Press **`+`** / **`-`** $\to$ Interactive zoom in / out
- **3D Ground Grid & Depth Perception:** Adaptive floor contact plane with metric grid squares, RGB coordinate orientation compass, and depth-sorted bone/joint rendering (Painter's algorithm).
- **Safety Feedback:** The 3D skeleton dynamically shifts from cyan/orange to bright red during confirmed fall emergencies.

---

## 🏗️ System Architecture

<p align="center">
  <img src="assets/pipeline_architecture.png" alt="End-to-end system pipeline diagram" width="380">
</p>

| Stage | Module | Responsibility |
|---|---|---|
| **Detection & Tracking** | `yolo_detector.py`, `person_tracker.py` | Detect people, assign persistent ByteTrack IDs, maintain per-person states |
| **Pose Estimation** | `pose_estimator.py` | Extract 33-point 2D pixel coordinates and metric 3D world landmarks |
| **Feature Extraction** | `feature_extractor.py` | Compute 20 biomechanical signals (angles, velocities, collapses, tilts) |
| **Fall Scoring** | `weighted_scoring.py` | Compute multi-factor weighted fall probability and severity grading |
| **Temporal State Machine** | `temporal_state_machine.py` | Temporal confirmation window and dual-path geometry reset |
| **Activity Monitoring** | `activity_monitor.py` | 3D/2D joint angle analysis, ADL classification, and duration analytics |
| **3D Visualization** | `visualizer_3d.py` | 3D perspective projection, mouse-orbit camera, and depth-sorted skeleton |
| **HUD & Dashboard** | `dashboard.py` | Combined HUD overlay, activity badges, signal factor panels, and fall banners |
| **Alerting & Logging** | `alarm_system.py`, `fall_logger.py` | Two-phase escalating audio alarm, CSV event logging, and snapshot capture |
| **Pipeline Orchestrator** | `fall_detector.py`, `main.py` | Synchronous dual-panel pipeline wiring, CLI options, and event handling |

---

## 📊 Benchmark Results (UR Fall Detection Dataset)

Evaluated on the **UR Fall Detection Dataset (URFD)** — 30 fall clips + 30 Activities of Daily Living (ADL) clips:

| Metric | Value |
|---|---|
| Sensitivity (Recall) | **93.33%** |
| Precision | 80.00% |
| F1 Score | 86.15% |
| Accuracy | 85.00% |
| Specificity | 76.67% |
| False Alarm Rate | 23.33% |
| Throughput | 25–30 FPS (CPU only on standard laptop) |

**Confusion Matrix**

| | Predicted: FALL | Predicted: NO FALL |
|---|---|---|
| **Actual: FALL** | TP = 28 | FN = 2 |
| **Actual: NO FALL** | FP = 7 | TN = 23 |

The system prioritizes sensitivity over precision by design — in health and safety monitoring, a missed fall is far more critical than an occasional transient alert.

---

## 📦 Project Structure

```
ICG-Project/
├── main.py                    # Application entrypoint (CLI / GUI launcher)
├── fall_detector.py           # Pipeline orchestrator (wires 2D + 3D + Activity)
├── activity_monitor.py        # Posture recognition & continuous ADL duration tracker
├── visualizer_3d.py           # Interactive 3D skeleton visualizer & orbit controls
├── dashboard.py                # HUD overlay, activity badges & factor panel
├── yolo_detector.py           # YOLOv8 + ByteTrack person detection
├── person_tracker.py          # Per-person state store, track inheritance & NMS
├── pose_estimator.py          # MediaPipe Pose Landmarker wrapper (2D + 3D metric)
├── feature_extractor.py       # 20 biomechanical signal computations
├── weighted_scoring.py        # 14-signal weighted fall scoring + severity
├── temporal_state_machine.py  # Temporal confirmation & reset state machine
├── alarm_system.py            # Two-phase audio alarm escalation
├── fall_logger.py             # CSV incident logging & screenshot persistence
├── test_activity_monitor.py   # Unit tests for posture recognition
├── test_visualizer_3d.py       # Unit tests for 3D projection & orbit controls
├── test_integration_step3.py   # End-to-end pipeline integration tests
├── test_edge_cases.py         # Stress and singularity edge case test suite
├── yolov8n.pt                  # YOLOv8 Nano weights (COCO-pretrained)
├── pose_landmarker.task        # MediaPipe Pose Landmarker model
├── assets/                     # Demo visuals and diagrams
├── docs/
│   └── report.pdf              # Full diploma project report
└── fall_logs/ · fall_events.csv # Generated at runtime on confirmed falls
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Nabyendu-the-noob/real-time-fall-detection.git
cd real-time-fall-detection

# Activate existing virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies (if setting up fresh)
pip install -r requirements.txt
```

### 2. Execution

```bash
# Run with pre-recorded video (Default: full dual-panel with 3D skeleton)
python main.py --source output_fall_detected_weighted.mp4

# Run with live webcam
python main.py --source 0

# Run in headless mode (server recording without GUI preview)
python main.py --source input.mp4 --no-display --output result_3d.mp4

# Optional: Run in legacy 2D-only mode
python main.py --source input.mp4 --no-3d
```

### 3. Interactive 3D Viewport Controls

When the application window is running:
* **Rotate in 3D:** Click and drag with the **Left Mouse Button** anywhere inside the right-hand 3D viewport.
* **Zoom:** Scroll the **Mouse Wheel** or hold **Right Mouse Button** and drag up/down.
* **Reset Camera:** **Double-Click** anywhere on the 3D viewport or press **`r`**.
* **Switch Views:** Press **`f`** (Front), **`s`** (Side), **`t`** (Top), or **`i`** (Isometric).
* **Quit:** Press **`q`** to safely close the application, save the video, and flush logs.

---

## 🧪 Verification & Testing

The project includes an automated test suite covering all modules, math operations, and edge cases:

```bash
python -m unittest discover -p "test_*.py"
```

**Results:**
```text
Ran 30 tests in 4.702s

OK
```
Tests validate 3D rotation matrix orthogonality, perspective projection, extreme zoom limits, `NaN`/`Inf` handling, posture classification, debounce windows, and end-to-end video processing.

---

## 📄 Citation

```bibtex
@article{Adhikary2026FallActivity3D,
  title={AI-Based Fall Detection and Activity Monitoring Using Pose Estimation with Interactive 3D Skeleton Visualization},
  author={Adhikary, Nabyendu and Halder, Sagar and Mandal, Prasenjit},
  advisor={Dr. Ajit Kumar Singh Yadav},
  journal={Diploma Final Year Project Report},
  institution={Ghani Khan Choudhury Institute of Engineering & Technology},
  year={2026}
}
```

## 🛠️ Tech Stack

Python 3.10+ · OpenCV 4.x · [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · ByteTrack · [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) · NumPy · SciPy · Pygame

## 📜 License

Released under the [MIT License](LICENSE).
