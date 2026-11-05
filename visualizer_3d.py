"""
Interactive 3D Skeleton Visualization Module
Layer: 3D Biomolecular & Kinematic Visualization

Renders real-time 3D skeleton pose estimations extracted from MediaPipe world landmarks
(metric 3D coordinates in meters, hip-centered).
Features:
- True interactive mouse orbit controls (Left Click + Drag to rotate yaw/pitch)
- Mouse wheel / Right Click + Drag to zoom in/out
- Double-click or 'R' key to reset camera to default perspective
- 3D perspective projection with depth-sorting (Painter's algorithm)
- 3D ground plane grid with metric depth perception
- Distinct color-coded left/right kinematic chains
- Real-time rendering at 30+ FPS directly in the display stream
- Optional standalone Matplotlib 3D viewer support
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# Kinematic bone connections in MediaPipe Pose with anatomical neck (33) and pelvis (34)
SKELETON_CONNECTIONS = [
    # ── Torso & Spine ──
    (33, 34, "spine"),       # Central Spine: Neck (33) -> Pelvis (34)
    (11, 33, "torso"),       # Left Clavicle: Left Shoulder (11) -> Neck (33)
    (12, 33, "torso"),       # Right Clavicle: Right Shoulder (12) -> Neck (33)
    (11, 12, "torso"),       # Shoulder girdle: Left Shoulder -> Right Shoulder
    (23, 34, "torso"),       # Left Pelvis: Left Hip (23) -> Pelvis (34)
    (24, 34, "torso"),       # Right Pelvis: Right Hip (24) -> Pelvis (34)
    (23, 24, "torso"),       # Pelvic basin: Left Hip -> Right Hip
    (11, 23, "torso"),       # Left Flank: Shoulder (11) -> Hip (23)
    (12, 24, "torso"),       # Right Flank: Shoulder (12) -> Hip (24)

    # ── Head & Neck ──
    (33, 0, "head"),         # Neck base (33) -> Nose (0)
    (0, 2, "head"),          # Nose (0) -> Left Eye (2)
    (2, 7, "head"),          # Left Eye (2) -> Left Ear (7)
    (0, 5, "head"),          # Nose (0) -> Right Eye (5)
    (5, 8, "head"),          # Right Eye (5) -> Right Ear (8)
    (7, 33, "head"),         # Left Jawline: Left Ear (7) -> Neck (33)
    (8, 33, "head"),         # Right Jawline: Right Ear (8) -> Neck (33)

    # ── Left Arm ──
    (11, 13, "left"),        # Left Upper Arm: Shoulder (11) -> Elbow (13)
    (13, 15, "left"),        # Left Forearm: Elbow (13) -> Wrist (15)
    (15, 17, "left"),        # Left Wrist (15) -> Pinky (17)
    (17, 19, "left"),        # Left Pinky (17) -> Index (19)
    (19, 15, "left"),        # Left Palm: Index (19) -> Wrist (15)
    (15, 21, "left"),        # Left Thumb: Wrist (15) -> Thumb (21)

    # ── Right Arm ──
    (12, 14, "right"),       # Right Upper Arm: Shoulder (12) -> Elbow (14)
    (14, 16, "right"),       # Right Forearm: Elbow (14) -> Wrist (16)
    (16, 18, "right"),       # Right Wrist (16) -> Pinky (18)
    (18, 20, "right"),       # Right Pinky (18) -> Index (20)
    (20, 16, "right"),       # Right Palm: Index (20) -> Wrist (16)
    (16, 22, "right"),       # Right Thumb: Wrist (16) -> Thumb (22)

    # ── Left Leg ──
    (23, 25, "left"),        # Left Thigh (Femur): Hip (23) -> Knee (25)
    (25, 27, "left"),        # Left Shin (Tibia): Knee (25) -> Ankle (27)
    (27, 29, "left"),        # Left Heel: Ankle (27) -> Heel (29)
    (29, 31, "left"),        # Left Sole: Heel (29) -> Foot Index (31)
    (27, 31, "left"),        # Left Foot Arch: Ankle (27) -> Foot Index (31)

    # ── Right Leg ──
    (24, 26, "right"),       # Right Thigh (Femur): Hip (24) -> Knee (26)
    (26, 28, "right"),       # Right Shin (Tibia): Knee (26) -> Ankle (28)
    (28, 30, "right"),       # Right Heel: Ankle (28) -> Heel (30)
    (30, 32, "right"),       # Right Sole: Heel (30) -> Foot Index (32)
    (28, 32, "right"),       # Right Foot Arch: Ankle (28) -> Foot Index (32)
]

# Color palette for 3D skeleton (BGR format)
PALETTE = {
    "left": (255, 195, 45),           # Vibrant Cyan / Sky Blue
    "right": (35, 145, 255),          # Vibrant Warm Orange
    "torso": (245, 245, 250),         # Pure White
    "spine": (255, 255, 255),         # Luminescent White
    "head": (85, 235, 115),           # Vibrant Lime Green
    "fall": (50, 50, 255),            # Emergency Red

    # Joint Colors (Outer / Inner Core)
    "joint_left": (255, 215, 90),
    "joint_left_core": (255, 245, 180),
    "joint_right": (60, 165, 255),
    "joint_right_core": (180, 220, 255),
    "joint_torso": (255, 255, 255),
    "joint_torso_core": (255, 255, 255),
    "joint_core": (255, 255, 255),
    "joint_head": (95, 245, 125),
    "joint_head_core": (200, 255, 210),
    "joint_fall": (70, 70, 255),
    "joint_fall_core": (180, 180, 255),

    "grid": (68, 76, 90),             # Visible Refined Slate Gray
    "grid_major": (110, 125, 145),    # Distinct Axis/Border
    "axis_x": (60, 60, 240),          # Red (X-axis)
    "axis_y": (60, 240, 60),          # Green (Y-axis)
    "axis_z": (240, 150, 60),         # Blue (Z-axis)
    "bg_top": (18, 20, 24),           # Gradient background top
    "bg_bottom": (30, 34, 42),        # Gradient background bottom
}

# Anatomical visual hierarchy for joint sizing (scale multiplier)
JOINT_TIER_SCALE = {
    # Tier 1: Major Articulations (1.35x)
    11: 1.35, 12: 1.35,  # Shoulders
    23: 1.35, 24: 1.35,  # Hips
    25: 1.35, 26: 1.35,  # Knees
    34: 1.30,            # Pelvis / Sacrum

    # Tier 2: Secondary Articulations (1.18x)
    13: 1.18, 14: 1.18,  # Elbows
    15: 1.18, 16: 1.18,  # Wrists
    27: 1.18, 28: 1.18,  # Ankles
    33: 1.15,            # Neck

    # Tier 3: Core & Facial Keypoints (1.0x)
    0: 1.0, 7: 1.0, 8: 1.0, 2: 0.9, 5: 0.9,

    # Tier 4: Extremities (Fingers & Toes) (0.72x)
    17: 0.72, 18: 0.72, 19: 0.72, 20: 0.72, 21: 0.72, 22: 0.72,
    29: 0.85, 30: 0.85, 31: 0.80, 32: 0.80,
}


def _get_bone_base_thickness(i1: int, i2: int, group: str) -> int:
    """Returns anatomical base thickness for a bone segment in pixels."""
    if group == "spine":
        return 5
    # Thighs / Femur (Hip -> Knee)
    if (i1 in (23, 24) and i2 in (25, 26)) or (i2 in (23, 24) and i1 in (25, 26)):
        return 5
    # Shins / Tibia (Knee -> Ankle)
    if (i1 in (25, 26) and i2 in (27, 28)) or (i2 in (25, 26) and i1 in (27, 28)):
        return 4
    # Upper Arms (Shoulder -> Elbow)
    if (i1 in (11, 12) and i2 in (13, 14)) or (i2 in (11, 12) and i1 in (13, 14)):
        return 4
    # Forearms (Elbow -> Wrist)
    if (i1 in (13, 14) and i2 in (15, 16)) or (i2 in (13, 14) and i1 in (15, 16)):
        return 4
    # Clavicles (Shoulder -> Neck)
    if (i1 in (11, 12) and i2 == 33) or (i2 in (11, 12) and i1 == 33):
        return 4
    # Pelvic girdle (Hip -> Pelvis)
    if (i1 in (23, 24) and i2 == 34) or (i2 in (23, 24) and i1 == 34):
        return 4
    # Torso flanks & girdles
    if (i1, i2) in ((11, 23), (12, 24), (23, 11), (24, 12)):
        return 3
    if (i1, i2) in ((11, 12), (12, 11), (23, 24), (24, 23)):
        return 3
    # Neck column
    if (i1 == 33 and i2 == 0) or (i2 == 33 and i1 == 0):
        return 4
    # Feet
    if i1 in (27, 28) or i2 in (29, 30, 31, 32):
        return 3
    # Hands / Fingers
    if i1 in (15, 16) or i2 in (17, 18, 19, 20, 21, 22):
        return 2
    # Facial contour
    if group == "head":
        return 2
    return 3


@dataclass
class CameraState:
    """Stores 3D orbit camera parameters."""
    azimuth: float = 25.0       # Yaw angle in degrees (-180 to 180)
    elevation: float = 12.0     # Pitch angle in degrees (-85 to 85)
    zoom: float = 1.0           # Zoom scaling factor (0.3 to 3.5)
    focal_length: float = 480.0 # Perspective focal length
    center_x: float = 0.0       # Dynamic center offset X (meters)
    center_y: float = -0.15     # Dynamic center offset Y (meters)
    center_z: float = 0.0       # Dynamic center offset Z (meters)
    cam_dist: float = 2.2       # Dynamic camera distance (meters)

    @property
    def center_y_offset(self) -> float:
        """Backwards compatibility alias for center_y."""
        return self.center_y

    @center_y_offset.setter
    def center_y_offset(self, val: float) -> None:
        self.center_y = val


class SkeletonVisualizer3D:
    """
    High-performance interactive 3D skeleton visualizer.
    Directly renders into a 2D BGR image canvas with full mouse orbit interactivity.
    """

    def __init__(
        self,
        width: int = 540,
        height: int = 720,
        enable_grid: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.enable_grid = enable_grid

        # Camera & interaction state
        self.camera = CameraState()
        self.is_dragging = False
        self.drag_button = None
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Viewport layout offset when attached side-by-side with main video
        self.x_offset = 0

        # Floor level tracker (in meters below hip center)
        self.floor_y_m = 0.95

        # Smoothing filters for automatic centering & dynamic camera distance
        self._smooth_center: Optional[np.ndarray] = None
        self._smooth_cam_dist: Optional[float] = None
        self.target_coverage: float = 0.62  # Skeleton occupies ~55-70% of viewport height

        # Performance optimization caches (background gradient & vectorized grid)
        self._cached_bg: Optional[np.ndarray] = None
        self._cached_bg_shape: Optional[Tuple[int, int]] = None
        self._grid_lines_base: Optional[np.ndarray] = None
        self._grid_is_major: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────
    # Camera Reset & Adjustments
    # ─────────────────────────────────────────────────────────────

    def reset_camera(self) -> None:
        """Resets camera to standard isometric viewpoint and auto-frames."""
        self.camera.azimuth = 25.0
        self.camera.elevation = 12.0
        self.camera.zoom = 1.0
        self._smooth_center = None
        self._smooth_cam_dist = None

    def set_preset_view(self, preset: str) -> None:
        """Sets standard orthographic/perspective camera angles."""
        preset = preset.lower()
        if preset in ("front", "f"):
            self.camera.azimuth = 0.0
            self.camera.elevation = 0.0
        elif preset in ("side", "s", "right"):
            self.camera.azimuth = 90.0
            self.camera.elevation = 0.0
        elif preset in ("left", "l"):
            self.camera.azimuth = -90.0
            self.camera.elevation = 0.0
        elif preset in ("top", "t"):
            self.camera.azimuth = 0.0
            self.camera.elevation = 85.0
        elif preset in ("iso", "isometric", "i"):
            self.reset_camera()

    def handle_key(self, key: int) -> bool:
        """
        Handles interactive keyboard shortcuts.
        'r' -> Reset camera
        'f' -> Front view
        's' -> Side view
        't' -> Top view
        'i' -> Isometric view
        '+'/'=' -> Zoom in
        '-' -> Zoom out
        Returns True if a shortcut was handled.
        """
        if key < 0:
            return False
        char = chr(key & 0xFF).lower()
        if char == "r":
            self.reset_camera()
            return True
        elif char == "f":
            self.set_preset_view("front")
            return True
        elif char == "s":
            self.set_preset_view("side")
            return True
        elif char == "t":
            self.set_preset_view("top")
            return True
        elif char == "i":
            self.set_preset_view("iso")
            return True
        elif char in ("+", "="):
            self.camera.zoom = min(3.5, self.camera.zoom * 1.15)
            return True
        elif char in ("-", "_"):
            self.camera.zoom = max(0.3, self.camera.zoom * 0.85)
            return True
        return False

    # ─────────────────────────────────────────────────────────────
    # Mouse Interaction Callbacks
    # ─────────────────────────────────────────────────────────────

    def attach_to_window(self, window_name: str, x_offset: int = 0) -> None:
        """Registers OpenCV mouse callback for the window."""
        self.x_offset = x_offset
        cv2.setMouseCallback(window_name, self._mouse_event_callback)

    def _mouse_event_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Internal OpenCV mouse listener."""
        self.handle_mouse(event, x, y, flags)

    def handle_mouse(self, event: int, x: int, y: int, flags: int = 0) -> bool:
        """
        Handles mouse interaction events.
        Coordinates are checked relative to the 3D viewport canvas.
        Returns True if the event was captured inside the 3D viewport.
        """
        local_x = x - self.x_offset
        local_y = y

        # Check if cursor is inside 3D viewport
        is_inside = (0 <= local_x < self.width) and (0 <= local_y < self.height)

        if event == cv2.EVENT_LBUTTONDOWN and is_inside:
            self.is_dragging = True
            self.drag_button = "left"
            self.last_mouse_x = local_x
            self.last_mouse_y = local_y
            return True

        elif event == cv2.EVENT_RBUTTONDOWN and is_inside:
            self.is_dragging = True
            self.drag_button = "right"
            self.last_mouse_x = local_x
            self.last_mouse_y = local_y
            return True

        elif event == cv2.EVENT_LBUTTONDBLCLK and is_inside:
            self.reset_camera()
            return True

        elif event == cv2.EVENT_MOUSEMOVE and self.is_dragging:
            dx = local_x - self.last_mouse_x
            dy = local_y - self.last_mouse_y
            self.last_mouse_x = local_x
            self.last_mouse_y = local_y

            if self.drag_button == "left":
                # Orbit rotation
                self.camera.azimuth = (self.camera.azimuth + dx * 0.65 + 180.0) % 360.0 - 180.0
                self.camera.elevation = max(-85.0, min(85.0, self.camera.elevation - dy * 0.65))
            elif self.drag_button == "right":
                # Drag zoom
                zoom_factor = 1.0 - (dy * 0.008)
                self.camera.zoom = max(0.3, min(3.5, self.camera.zoom * zoom_factor))
            return True

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.is_dragging = False
            self.drag_button = None
            return True

        # Mouse wheel zoom support (cv2.EVENT_MOUSEWHEEL)
        elif event == 10 and is_inside:  # 10 is EVENT_MOUSEWHEEL on Windows OpenCV
            if flags > 0:
                self.camera.zoom = min(3.5, self.camera.zoom * 1.1)
            else:
                self.camera.zoom = max(0.3, self.camera.zoom * 0.9)
            return True

        return False

    # ─────────────────────────────────────────────────────────────
    # 3D Math, Rotation & Perspective Projection
    # ─────────────────────────────────────────────────────────────

    def _get_rotation_matrix(self) -> np.ndarray:
        """Computes combined 3D camera rotation matrix Rx(elevation) * Ry(azimuth)."""
        az_rad = math.radians(self.camera.azimuth)
        el_rad = math.radians(self.camera.elevation)

        # Yaw around Y-axis
        cos_y, sin_y = math.cos(az_rad), math.sin(az_rad)
        R_y = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=np.float32)

        # Pitch around X-axis
        cos_x, sin_x = math.cos(el_rad), math.sin(el_rad)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x]
        ], dtype=np.float32)

        return R_x @ R_y

    def _project_point(
        self, pt_3d: np.ndarray, rot_matrix: np.ndarray, cam_dist: Optional[float] = None
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Projects a 3D point (x, y, z) into 2D viewport (u, v) with perspective depth.
        pt_3d is in meters (+Y upwards).
        Returns ((u, v), depth_z).
        Robust against NaNs, Infs, and extreme projections.
        """
        # Guard against NaN/Inf
        if not np.all(np.isfinite(pt_3d)):
            return None, 0.0

        # Apply camera center offset (automatically centers skeleton)
        shifted = np.array([
            pt_3d[0] - self.camera.center_x,
            pt_3d[1] - self.camera.center_y,
            pt_3d[2] - self.camera.center_z,
        ], dtype=np.float32)

        # Rotate point
        rotated = rot_matrix @ shifted

        # Perspective camera position
        dist = float(cam_dist if cam_dist is not None else self.camera.cam_dist)
        depth_z = float(rotated[2] + dist)

        if depth_z <= 0.2:
            return None, depth_z

        focal = self.camera.focal_length * self.camera.zoom
        cx = self.width / 2.0
        cy = self.height / 2.0

        raw_u = cx + (rotated[0] / depth_z) * focal
        raw_v = cy - (rotated[1] / depth_z) * focal

        if not (math.isfinite(raw_u) and math.isfinite(raw_v)):
            return None, depth_z

        u = int(max(-3000, min(3000, raw_u)))
        v = int(max(-3000, min(3000, raw_v)))

        return (u, v), depth_z

    def _project_points_batch(
        self, pts_3d: np.ndarray, rot_matrix: np.ndarray, cam_dist: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized 3D perspective projection for an (N, 3) batch of points in meters.
        Returns:
            u: (N,) int32 screen X coordinates
            v: (N,) int32 screen Y coordinates
            depth_z: (N,) float32 camera depth
            valid: (N,) bool mask indicating points in front of camera and within bounds
        """
        dist = float(cam_dist if cam_dist is not None else self.camera.cam_dist)
        focal = self.camera.focal_length * self.camera.zoom
        cx = self.width / 2.0
        cy = self.height / 2.0

        # Center shift with NaN/Inf finite guard
        center = np.array(
            [self.camera.center_x, self.camera.center_y, self.camera.center_z],
            dtype=np.float32,
        )
        shifted = pts_3d - center
        finite_mask = np.all(np.isfinite(shifted), axis=1)
        safe_shifted = np.where(finite_mask[:, None], shifted, 0.0)

        # Batch rotation: (N, 3) @ (3, 3).T
        rotated = safe_shifted @ rot_matrix.T
        depth_z = np.where(finite_mask, rotated[:, 2] + dist, -999.0)

        valid = finite_mask & (depth_z > 0.2)

        safe_z = np.where(valid, depth_z, 1.0)
        safe_rot_x = np.where(valid, rotated[:, 0], 0.0)
        safe_rot_y = np.where(valid, rotated[:, 1], 0.0)

        raw_u = cx + (safe_rot_x / safe_z) * focal
        raw_v = cy - (safe_rot_y / safe_z) * focal

        valid = valid & np.isfinite(raw_u) & np.isfinite(raw_v)
        u = np.clip(np.nan_to_num(raw_u, nan=0.0), -3000, 3000).astype(np.int32)
        v = np.clip(np.nan_to_num(raw_v, nan=0.0), -3000, 3000).astype(np.int32)

        return u, v, depth_z, valid

    # ─────────────────────────────────────────────────────────────
    # Canvas Background & Ground Grid Rendering
    # ─────────────────────────────────────────────────────────────

    def _ensure_background_cached(self) -> np.ndarray:
        """Maintains pre-rendered gradient background for ultra-low latency copying."""
        shape = (self.height, self.width)
        if self._cached_bg is None or self._cached_bg_shape != shape:
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            c_top = np.array(PALETTE["bg_top"], dtype=np.float32)
            c_bot = np.array(PALETTE["bg_bottom"], dtype=np.float32)

            for i in range(self.height):
                t = i / float(self.height)
                color = (1.0 - t) * c_top + t * c_bot
                canvas[i, :] = color.astype(np.uint8)

            self._cached_bg = canvas
            self._cached_bg_shape = shape
            self._build_grid_template()

        return self._cached_bg.copy()

    def _create_background(self) -> np.ndarray:
        """Creates a modern vertical gradient studio backdrop (cached for performance)."""
        return self._ensure_background_cached()

    def _build_grid_template(self) -> None:
        """Precomputes ground grid vertex coordinates for batch projection."""
        grid_range = 1.25  # +/- 1.25 meters (2.5m x 2.5m grid)
        step = 0.25        # 25 cm per grid square

        x_vals = np.arange(-grid_range, grid_range + step * 0.5, step)
        z_vals = np.arange(-grid_range, grid_range + step * 0.5, step)

        pts = []
        is_major = []

        # X lines (parallel to Z)
        for gx in x_vals:
            pts.append([gx, 0.0, -grid_range])
            pts.append([gx, 0.0, grid_range])
            is_major.append(abs(gx) < 1e-4)

        # Z lines (parallel to X)
        for gz in z_vals:
            pts.append([-grid_range, 0.0, gz])
            pts.append([grid_range, 0.0, gz])
            is_major.append(abs(gz) < 1e-4)

        self._grid_lines_base = np.array(pts, dtype=np.float32)
        self._grid_is_major = np.array(is_major, dtype=bool)

    def _draw_ground_grid(self, canvas: np.ndarray, rot_matrix: np.ndarray) -> None:
        """Renders an interactive 3D ground grid plane with vectorized projection."""
        floor_y = -self.floor_y_m
        if self._grid_lines_base is None:
            self._build_grid_template()

        # Update floor Y coordinate on cached vertices
        grid_pts = self._grid_lines_base.copy()
        grid_pts[:, 1] = floor_y

        u, v, _, valid = self._project_points_batch(grid_pts, rot_matrix)

        n_lines = len(self._grid_is_major)
        for i in range(n_lines):
            p1_idx = i * 2
            p2_idx = p1_idx + 1
            if valid[p1_idx] and valid[p2_idx]:
                is_major = bool(self._grid_is_major[i])
                color = PALETTE["grid_major"] if is_major else PALETTE["grid"]
                thickness = 2 if is_major else 1
                cv2.line(
                    canvas,
                    (int(u[p1_idx]), int(v[p1_idx])),
                    (int(u[p2_idx]), int(v[p2_idx])),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        # Coordinate axes at floor origin
        axes_pts = np.array([
            [0.0, floor_y, 0.0],
            [0.30, floor_y, 0.0],
            [0.0, floor_y + 0.30, 0.0],
            [0.0, floor_y, 0.30],
        ], dtype=np.float32)

        au, av, _, avalid = self._project_points_batch(axes_pts, rot_matrix)
        if avalid[0]:
            po = (int(au[0]), int(av[0]))
            if avalid[1]:
                cv2.line(canvas, po, (int(au[1]), int(av[1])), PALETTE["axis_x"], 2, cv2.LINE_AA)
            if avalid[2]:
                cv2.line(canvas, po, (int(au[2]), int(av[2])), PALETTE["axis_y"], 2, cv2.LINE_AA)
            if avalid[3]:
                cv2.line(canvas, po, (int(au[3]), int(av[3])), PALETTE["axis_z"], 2, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────
    # Main Skeleton Rendering (Depth Sorted)
    # ─────────────────────────────────────────────────────────────

    def render(
        self,
        world_landmarks: Optional[List[Any]],
        activity_label: Optional[str] = None,
        confidence: Optional[float] = None,
        is_fall: bool = False,
    ) -> np.ndarray:
        """
        Renders the interactive 3D skeleton frame with vectorized math and depth sorting.
        world_landmarks: List of 33 MediaPipe world landmark objects with x, y, z in meters.
        Returns BGR numpy image of shape (height, width, 3).
        """
        canvas = self._create_background()
        rot_matrix = self._get_rotation_matrix()

        if not world_landmarks or len(world_landmarks) < 33:
            if self.enable_grid:
                self._draw_ground_grid(canvas, rot_matrix)
            # Standby watermark when no pose is visible
            cv2.putText(
                canvas,
                "WAITING FOR 3D POSE...",
                (self.width // 2 - 120, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (90, 105, 120),
                1,
                cv2.LINE_AA,
            )
            self._draw_overlay_hud(canvas, activity_label, confidence, is_fall)
            if canvas.shape[1] > 2:
                canvas[:, 0:1] = (42, 38, 52)
                canvas[:, 1:2] = (68, 66, 82)
            return canvas

        # Vectorized landmark extraction with anatomical virtual keypoints
        n_raw = min(33, len(world_landmarks))
        # Total points: 33 raw landmarks + 2 virtual (33: Neck, 34: Pelvis)
        n_total = 35
        pts_arr = np.empty((n_total, 3), dtype=np.float32)
        vis_arr = np.empty(n_total, dtype=np.float32)

        for i in range(n_raw):
            lm = world_landmarks[i]
            pts_arr[i, 0] = lm.x
            pts_arr[i, 1] = -lm.y
            pts_arr[i, 2] = lm.z
            vis_arr[i] = getattr(lm, "visibility", 1.0)

        # Guard for any missing raw landmarks
        for i in range(n_raw, 33):
            pts_arr[i] = [0.0, 0.0, 0.0]
            vis_arr[i] = 0.0

        # Virtual Neck (33): anatomical midpoint between Left & Right Shoulders
        if vis_arr[11] > 0.20 and vis_arr[12] > 0.20:
            pts_arr[33] = 0.5 * (pts_arr[11] + pts_arr[12])
            vis_arr[33] = min(vis_arr[11], vis_arr[12])
        else:
            pts_arr[33] = pts_arr[11] if vis_arr[11] > 0.20 else pts_arr[12]
            vis_arr[33] = max(vis_arr[11], vis_arr[12]) * 0.7

        # Virtual Pelvis (34): anatomical midpoint between Left & Right Hips
        if vis_arr[23] > 0.20 and vis_arr[24] > 0.20:
            pts_arr[34] = 0.5 * (pts_arr[23] + pts_arr[24])
            vis_arr[34] = min(vis_arr[23], vis_arr[24])
        else:
            pts_arr[34] = pts_arr[23] if vis_arr[23] > 0.20 else pts_arr[24]
            vis_arr[34] = max(vis_arr[23], vis_arr[24]) * 0.7

        # Filter valid/visible landmarks for automatic centering & dynamic framing (raw 33 landmarks)
        valid_mask_raw = (vis_arr[:33] > 0.25) & np.all(np.isfinite(pts_arr[:33]), axis=1)
        valid_pts = pts_arr[:33][valid_mask_raw]

        if len(valid_pts) >= 4:
            x_min, y_min, z_min = np.min(valid_pts, axis=0)
            x_max, y_max, z_max = np.max(valid_pts, axis=0)

            span_x = max(0.20, float(x_max - x_min))
            span_y = max(0.20, float(y_max - y_min))
            span_z = max(0.20, float(z_max - z_min))

            # Horizontal and depth center: midpoint of detected bounding box
            center_x = float((x_min + x_max) / 2.0)
            center_z = float((z_min + z_max) / 2.0)

            # Vertical center: blend bounding box midpoint and hip center
            bbox_center_y = float((y_min + y_max) / 2.0)
            if (len(world_landmarks) > 24 and
                vis_arr[23] > 0.3 and
                vis_arr[24] > 0.3):
                hip_mid_y = float((pts_arr[23, 1] + pts_arr[24, 1]) / 2.0)
                raw_center_y = 0.75 * bbox_center_y + 0.25 * hip_mid_y
            else:
                raw_center_y = bbox_center_y

            # Shifting center_y slightly upward (+0.07m) places the visual center of the skeleton
            # directly in the clear viewing window below the top HUD activity badge
            target_center = np.array([center_x, raw_center_y + 0.07, center_z], dtype=np.float32)

            # Determine optimal camera distance to achieve 55%-70% viewport height coverage
            diag_span = float(math.sqrt(span_x * span_x + span_y * span_y + span_z * span_z))
            s_ref = max(span_y, 0.85 * max(span_x, span_z), 0.70 * diag_span)
            s_ref = max(0.85, min(2.5, s_ref))  # Anatomical clamp for human body dimensions

            focal = self.camera.focal_length
            target_coverage = self.target_coverage

            d_height = (focal * s_ref) / max(1.0, target_coverage * self.height)
            d_width = (focal * max(span_x, span_z)) / max(1.0, 0.78 * self.width)
            target_cam_dist = max(1.4, min(4.5, max(d_height, d_width)))

            # Smooth interpolation (EMA) to prevent visual jitter between frames
            if self._smooth_center is None:
                self._smooth_center = target_center.copy()
                self._smooth_cam_dist = target_cam_dist
            else:
                alpha_pos = 0.15
                alpha_dist = 0.10
                self._smooth_center = (1.0 - alpha_pos) * self._smooth_center + alpha_pos * target_center
                self._smooth_cam_dist = (1.0 - alpha_dist) * self._smooth_cam_dist + alpha_dist * target_cam_dist

            self.camera.center_x = float(self._smooth_center[0])
            self.camera.center_y = float(self._smooth_center[1])
            self.camera.center_z = float(self._smooth_center[2])
            self.camera.cam_dist = float(self._smooth_cam_dist)

        # Update floor level dynamically (lowest body contact point in meters)
        contact_mask = (vis_arr[:33] > 0.35) & np.isfinite(pts_arr[:33, 1])
        if np.any(contact_mask):
            detected_floor = float(np.max(-pts_arr[:33, 1][contact_mask]))  # Raw MediaPipe +Y downwards
            self.floor_y_m = 0.90 * self.floor_y_m + 0.10 * detected_floor

        # Draw ground grid with updated camera and center
        if self.enable_grid:
            self._draw_ground_grid(canvas, rot_matrix)

        # Vectorized projection of all 35 keypoints (33 landmarks + 2 virtual)
        u, v, depth_z, valid = self._project_points_batch(pts_arr, rot_matrix)

        # Collect 3D bones for depth sorting
        bones_to_render = []
        cam_zoom = self.camera.zoom
        for i1, i2, group in SKELETON_CONNECTIONS:
            if (i1 < n_total and i2 < n_total and
                valid[i1] and valid[i2] and
                vis_arr[i1] >= 0.25 and vis_arr[i2] >= 0.25):

                avg_z = float((depth_z[i1] + depth_z[i2]) / 2.0)
                color = PALETTE.get(group, PALETTE["torso"])
                if is_fall:
                    color = PALETTE["fall"]

                base_thick = _get_bone_base_thickness(i1, i2, group)
                depth_scale = (1.6 / max(0.5, avg_z)) * cam_zoom
                thick = max(1, min(7, int(base_thick * depth_scale)))

                bones_to_render.append((
                    avg_z,
                    "bone",
                    (int(u[i1]), int(v[i1])),
                    (int(u[i2]), int(v[i2])),
                    color,
                    thick,
                    group,
                ))

        # Collect 3D joint nodes for depth sorting with anatomical visual hierarchy
        joints_to_render = []
        for idx in range(n_total):
            if valid[idx] and vis_arr[idx] >= 0.25:
                # Color assignment matching kinematic chain
                if is_fall:
                    body_color = PALETTE["joint_fall"]
                    core_color = PALETTE["joint_fall_core"]
                elif idx in (11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31):
                    body_color = PALETTE["joint_left"]
                    core_color = PALETTE["joint_left_core"]
                elif idx in (12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
                    body_color = PALETTE["joint_right"]
                    core_color = PALETTE["joint_right_core"]
                elif idx in (33, 34):  # Spine column (Neck, Pelvis)
                    body_color = PALETTE["joint_torso"]
                    core_color = PALETTE["joint_torso_core"]
                elif idx <= 10:  # Head & facial
                    body_color = PALETTE["joint_head"]
                    core_color = PALETTE["joint_head_core"]
                else:
                    body_color = PALETTE["joint_core"]
                    core_color = PALETTE["joint_core"]

                z_val = float(depth_z[idx])
                tier_mult = JOINT_TIER_SCALE.get(idx, 0.72)
                base_r = 4.6 * (1.8 / max(0.4, z_val)) * cam_zoom
                radius = max(2, min(9, int(base_r * tier_mult)))

                # Give joints slight depth priority (-0.008) so they neatly cap bone endpoints
                effective_z = z_val - 0.008
                joints_to_render.append((
                    effective_z,
                    "joint",
                    (int(u[idx]), int(v[idx])),
                    radius,
                    body_color,
                    core_color,
                    tier_mult,
                ))

        # Sort all elements from back to front (Painter's algorithm)
        all_drawables = bones_to_render + joints_to_render
        all_drawables.sort(key=lambda item: item[0], reverse=True)

        # Draw sorted elements with anti-aliasing and depth highlights
        for item in all_drawables:
            elem_type = item[1]
            if elem_type == "bone":
                _, _, p1, p2, color, thick, group = item
                cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)
                # Volumetric inner core highlight for heavy bones
                if thick >= 4 and not is_fall:
                    highlight_col = tuple(min(255, int(c * 1.25 + 25)) for c in color)
                    cv2.line(canvas, p1, p2, highlight_col, max(1, thick - 2), cv2.LINE_AA)
            elif elem_type == "joint":
                _, _, center, radius, body_col, core_col, tier_mult = item
                # 1. Dark silhouette boundary halo for distinct joint definition
                cv2.circle(canvas, center, radius + 1, (14, 16, 20), -1, cv2.LINE_AA)
                # 2. Main spherical joint disc
                cv2.circle(canvas, center, radius, body_col, -1, cv2.LINE_AA)
                # 3. Luminous specular center highlight (Tier 1 & 2 articulations)
                if tier_mult >= 1.15 and radius >= 4:
                    cv2.circle(canvas, center, max(1, radius - 2), core_col, -1, cv2.LINE_AA)
                    cv2.circle(canvas, center, max(1, radius // 2), (255, 255, 255), -1, cv2.LINE_AA)

        # Draw HUD overlays and instructions
        self._draw_overlay_hud(canvas, activity_label, confidence, is_fall)

        # Clear visual boundary along left edge separating 2D camera feed from 3D viewport
        if canvas.shape[1] > 2:
            canvas[:, 0:1] = (42, 38, 52)
            canvas[:, 1:2] = (68, 66, 82)

        return canvas

    # ─────────────────────────────────────────────────────────────
    # Interactive HUD Overlays
    # ─────────────────────────────────────────────────────────────

    def _draw_overlay_hud(
        self,
        canvas: np.ndarray,
        activity: Optional[str],
        confidence: Optional[float],
        is_fall: bool,
    ) -> None:
        """Renders camera info, mouse tips, and active state pill."""
        # Top HUD Card with translucent dark backdrop
        card_w = min(340, self.width - 24)
        card_h = 76 if activity else 50
        cx1, cy1 = 14, 12
        cx2, cy2 = cx1 + card_w, cy1 + card_h

        if cx2 <= self.width and cy2 <= self.height and card_w > 50 and card_h > 20:
            roi = canvas[cy1:cy2, cx1:cx2]
            if roi.size > 0:
                cv2.addWeighted(roi, 0.22, np.full_like(roi, 20), 0.78, 0, roi)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (55, 52, 68), 1)

            # Top Title
            cv2.putText(
                canvas,
                "3D SKELETON VIEW",
                (cx1 + 10, cy1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (235, 240, 245),
                1,
                cv2.LINE_AA,
            )

            # Camera Orientation Readout
            cam_info = f"Az: {int(self.camera.azimuth)}° | El: {int(self.camera.elevation)}° | Zoom: {self.camera.zoom:.1f}x"
            cv2.putText(
                canvas,
                cam_info,
                (cx1 + 10, cy1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (148, 158, 172),
                1,
                cv2.LINE_AA,
            )

            # Activity Badge
            if activity:
                act_str = activity.upper()
                conf_str = f" — {int(confidence * 100)}%" if confidence is not None else ""
                act_text = f"● {act_str}{conf_str}"

                if is_fall:
                    badge_color = (40, 50, 240)    # Crimson Red
                elif activity == "Standing":
                    badge_color = (80, 210, 60)    # Green
                elif activity in ("Walking", "Bending"):
                    badge_color = (0, 215, 255)    # Yellow
                elif activity == "Sitting":
                    badge_color = (245, 160, 45)   # Cerulean Blue
                else:
                    badge_color = (200, 140, 255)

                cv2.putText(
                    canvas,
                    act_text,
                    (cx1 + 10, cy1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    badge_color,
                    1,
                    cv2.LINE_AA,
                )

        # Bottom help footer
        help_text = "[Left-Drag] Rotate  |  [Wheel / Right-Drag] Zoom  |  [Dbl-Click] Reset"
        cv2.putText(
            canvas,
            help_text,
            (16, self.height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (120, 130, 145),
            1,
            cv2.LINE_AA,
        )

