"""
PowerPoint Presentation Generator for AI Fall Detection & 3D Visualization Project.
Builds a professional 48-slide academic deck in 16:9 widescreen format using python-pptx.
Theme: Dark Workstation / Clinical AI Monitoring (Deep Slate Navy, Cyan & Emerald Accents).
"""

from __future__ import annotations

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE


# ─────────────────────────────────────────────────────────────────────────────
# Theme Color Palette (RGB)
# ─────────────────────────────────────────────────────────────────────────────
COLOR_BG_DARK       = RGBColor(11, 17, 30)       # #0B111E - Deep Slate Navy
COLOR_CARD_BG       = RGBColor(26, 35, 50)       # #1A2332 - Card Container
COLOR_CARD_BORDER   = RGBColor(45, 60, 85)       # #2D3C55 - Subtle Border
COLOR_ACCENT_CYAN   = RGBColor(56, 189, 248)     # #38BDF8 - AI / Vision Accent
COLOR_ACCENT_GREEN  = RGBColor(34, 197, 94)      # #22C55E - Clinical Normal / Recovery
COLOR_ACCENT_RED    = RGBColor(239, 68, 68)      # #EF4444 - Critical Fall / Emergency
COLOR_ACCENT_AMBER  = RGBColor(245, 158, 11)     # #F59E0B - Warning / Candidate
COLOR_ACCENT_MUTED  = RGBColor(148, 163, 184)    # #94A3B8 - Neutral / Muted Accent
COLOR_TEXT_WHITE    = RGBColor(248, 250, 252)    # #F8FAFC - Primary Title
COLOR_TEXT_LIGHT    = RGBColor(226, 232, 240)    # #E2E8F0 - High-Contrast Body
COLOR_TEXT_MUTED    = RGBColor(148, 163, 184)    # #94A3B8 - Subtitle / Footnote
COLOR_CALLOUT_BG    = RGBColor(18, 28, 42)       # #121C2A - Takeaway Bar


class SlideDeckGenerator:
    def __init__(self, output_path: str = "AI_Fall_Detection_Project_Presentation.pptx"):
        self.output_path = output_path
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.blank_layout = self.prs.slide_layouts[6]
        self.total_slides = 48
        self.current_slide_num = 0

    def _apply_background(self, slide):
        """Applies solid dark background rectangle across entire slide."""
        bg = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), self.prs.slide_width, self.prs.slide_height
        )
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG_DARK
        bg.line.fill.background()
        return bg

    def _add_header(self, slide, category: str, title: str):
        """Adds standard category badge, slide title, and accent rule."""
        self.current_slide_num += 1

        # Category Pill Badge
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(3.2), Inches(0.35)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = RGBColor(20, 32, 48)
        badge.line.color.rgb = COLOR_ACCENT_CYAN
        badge.line.width = Pt(1)
        tf_b = badge.text_frame
        tf_b.word_wrap = False
        tf_b.margin_left = Inches(0.1)
        tf_b.margin_top = Inches(0.02)
        p_b = tf_b.paragraphs[0]
        p_b.text = category.upper()
        p_b.font.size = Pt(10)
        p_b.font.bold = True
        p_b.font.color.rgb = COLOR_ACCENT_CYAN

        # Slide Title
        tb_t = slide.shapes.add_textbox(Inches(0.75), Inches(0.8), Inches(11.8), Inches(0.7))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_WHITE

        # Accent Rule
        line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.52), Inches(11.73), Inches(0.03)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_ACCENT_CYAN
        line.line.fill.background()

        # Footer
        footer_tb = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.73), Inches(0.35))
        tf_f = footer_tb.text_frame
        p_f = tf_f.paragraphs[0]
        p_f.text = f"AI Fall Detection & Activity Monitoring | 3D Skeleton Engine                                                     Slide {self.current_slide_num} of {self.total_slides}"
        p_f.font.size = Pt(9.5)
        p_f.font.color.rgb = COLOR_TEXT_MUTED

    def _add_takeaway_bar(self, slide, text: str, accent_color: RGBColor = COLOR_ACCENT_CYAN):
        """Adds a highlighted takeaway or key metric card at the bottom."""
        bar = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.15), Inches(11.73), Inches(0.75)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = COLOR_CALLOUT_BG
        bar.line.color.rgb = accent_color
        bar.line.width = Pt(1.5)

        tf = bar.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_top = Inches(0.1)
        p = tf.paragraphs[0]
        p.text = f"KEY TAKEAWAY / CORE PRINCIPLE: {text}"
        p.font.size = Pt(10.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE

    def add_title_slide(self):
        """Creates Slide 1: Primary Title Slide."""
        self.current_slide_num += 1
        slide = self.prs.slides.add_slide(self.blank_layout)
        self._apply_background(slide)

        # Center Card
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(0.9), Inches(10.93), Inches(5.7)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_ACCENT_CYAN
        card.line.width = Pt(2)

        # Badge
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.3), Inches(1.2), Inches(4.7), Inches(0.4)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = RGBColor(15, 24, 38)
        badge.line.color.rgb = COLOR_ACCENT_GREEN
        p_b = badge.text_frame.paragraphs[0]
        p_b.text = "ACADEMIC PROJECT PRESENTATION"
        p_b.alignment = PP_ALIGN.CENTER
        p_b.font.bold = True
        p_b.font.size = Pt(11)
        p_b.font.color.rgb = COLOR_ACCENT_GREEN

        # Title Textbox
        tb = slide.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.3), Inches(2.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = "AI-Based Fall Detection and Activity Monitoring Using Pose Estimation with Interactive 3D Skeleton Visualization"
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(27)
        p.font.bold = True
        p.font.color.rgb = COLOR_TEXT_WHITE

        # Subtitle
        p2 = tf.add_paragraph()
        p2.text = "\nA Real-Time Edge-Optimized Computer Vision System for Geriatric Healthcare and Autonomous Safety Telemetry"
        p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(15)
        p2.font.color.rgb = COLOR_ACCENT_CYAN

        # Author / Metadata Card
        meta_card = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(2.0), Inches(4.3), Inches(9.3), Inches(1.8)
        )
        meta_card.fill.solid()
        meta_card.fill.fore_color.rgb = RGBColor(18, 25, 38)
        meta_card.line.color.rgb = COLOR_CARD_BORDER
        tf_m = meta_card.text_frame
        tf_m.word_wrap = True
        
        pm1 = tf_m.paragraphs[0]
        pm1.text = "ENGINEERING STACK & CORE ATTRIBUTES"
        pm1.alignment = PP_ALIGN.CENTER
        pm1.font.bold = True
        pm1.font.size = Pt(12)
        pm1.font.color.rgb = COLOR_ACCENT_AMBER

        pm2 = tf_m.add_paragraph()
        pm2.text = "• YOLOv8 Nano Detection  |  ByteTrack Kalman Multi-Object Tracking  |  MediaPipe 3D Pose Estimation\n• 12-Factor Biomechanical Scoring Engine  |  Finite Temporal State Machine  |  Custom 3D Mol* Projection\n• Workstation Dual-Panel HUD  |  Non-Blocking Multi-Channel Alert Dispatch (Telegram Bot + SMTP Email)"
        pm2.alignment = PP_ALIGN.CENTER
        pm2.font.size = Pt(11)
        pm2.font.color.rgb = COLOR_TEXT_LIGHT

    def add_dual_card_slide(
        self,
        category: str,
        title: str,
        card1_title: str,
        card1_bullets: list[str],
        card2_title: str,
        card2_bullets: list[str],
        takeaway: str,
        card1_color: RGBColor = COLOR_ACCENT_CYAN,
        card2_color: RGBColor = COLOR_ACCENT_GREEN,
    ):
        """Creates a standardized 2-card comparison / parallel slide."""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self._apply_background(slide)
        self._add_header(slide, category, title)

        # Card 1 (Left)
        c1 = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.7), Inches(4.3)
        )
        c1.fill.solid()
        c1.fill.fore_color.rgb = COLOR_CARD_BG
        c1.line.color.rgb = COLOR_CARD_BORDER
        c1.line.width = Pt(1.5)

        tb1 = slide.shapes.add_textbox(Inches(0.9), Inches(1.8), Inches(5.5), Inches(4.1))
        tf1 = tb1.text_frame
        tf1.word_wrap = True
        p1 = tf1.paragraphs[0]
        p1.text = card1_title.upper()
        p1.font.bold = True
        p1.font.size = Pt(14)
        p1.font.color.rgb = card1_color

        for item in card1_bullets:
            p = tf1.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11)
            p.font.color.rgb = COLOR_TEXT_LIGHT
            p.space_after = Pt(4)

        # Card 2 (Right)
        c2 = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.83), Inches(1.7), Inches(5.7), Inches(4.3)
        )
        c2.fill.solid()
        c2.fill.fore_color.rgb = COLOR_CARD_BG
        c2.line.color.rgb = COLOR_CARD_BORDER
        c2.line.width = Pt(1.5)

        tb2 = slide.shapes.add_textbox(Inches(6.93), Inches(1.8), Inches(5.5), Inches(4.1))
        tf2 = tb2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = card2_title.upper()
        p2.font.bold = True
        p2.font.size = Pt(14)
        p2.font.color.rgb = card2_color

        for item in card2_bullets:
            p = tf2.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11)
            p.font.color.rgb = COLOR_TEXT_LIGHT
            p.space_after = Pt(4)

        self._add_takeaway_bar(slide, takeaway, card1_color)

    def add_wide_card_slide(
        self,
        category: str,
        title: str,
        card_title: str,
        bullets: list[str],
        takeaway: str,
        accent_color: RGBColor = COLOR_ACCENT_CYAN,
    ):
        """Creates a wide 1-card structured technical overview slide."""
        slide = self.prs.slides.add_slide(self.blank_layout)
        self._apply_background(slide)
        self._add_header(slide, category, title)

        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(11.73), Inches(4.3)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER
        card.line.width = Pt(1.5)

        tb = slide.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(11.33), Inches(4.0))
        tf = tb.text_frame
        tf.word_wrap = True
        p_head = tf.paragraphs[0]
        p_head.text = card_title.upper()
        p_head.font.bold = True
        p_head.font.size = Pt(15)
        p_head.font.color.rgb = accent_color

        for item in bullets:
            p = tf.add_paragraph()
            p.text = f"• {item}"
            p.font.size = Pt(11.5)
            p.font.color.rgb = COLOR_TEXT_LIGHT
            p.space_after = Pt(5)

        self._add_takeaway_bar(slide, takeaway, accent_color)

    def save(self):
        self.prs.save(self.output_path)
        print(f"[SUCCESS] Presentation generated successfully: {self.output_path} ({self.current_slide_num} slides)")


def build_all_48_slides():
    builder = SlideDeckGenerator("AI_Fall_Detection_Project_Presentation.pptx")

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 1: Title
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_title_slide()

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 2: Problem Context & Global Clinical Motivation
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 1: Clinical Foundations",
        title="Problem Context & Global Geriatric Health Impact",
        card1_title="Epidemiological Realities",
        card1_bullets=[
            "Second Leading Accidental Death Cause: WHO classifies unintentional falls as the #2 global cause of injury deaths.",
            "High Mortality in Elderly: Adults over 60 suffer the highest rates of fatal and debilitating fall incidents.",
            "Severe Morbidity Outcomes: 20–30% of falls cause moderate-to-severe injuries (hip fractures, subdural hematomas).",
            "Demographic Tsunami: Global population aged 60+ is expanding toward 1.6 billion by 2050, straining nursing capacity."
        ],
        card2_title="The Critical 'Long Lie' Complication",
        card2_bullets=[
            "Definition: Remaining prone and incapacitated on the floor for >1 hour following a collapse.",
            "Fatal Pathologies: Triggers acute hypothermia, rhabdomyolysis, renal failure, dehydration, and pressure ulcers.",
            "Mortality Multiplier: 50% of elderly fallers experiencing a long lie die within 6 months, even without primary trauma.",
            "Immediate Intervention Window: Discovery within 15 minutes reduces hospitalization and long-term care by 74%."
        ],
        takeaway="Immediate, autonomous detection without relying on the victim to press a button directly prevents fatal long-lie outcomes.",
        card1_color=COLOR_ACCENT_AMBER,
        card2_color=COLOR_ACCENT_RED
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 3: Critique of Existing Approaches
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 1: Clinical Foundations",
        title="Critique of Existing Systems: Wearables vs. Vision",
        card1_title="Wearables & Ambient Pressure Pads",
        card1_bullets=[
            "Human Non-Compliance: Forgotten on chargers, removed during showering or sleeping (high-risk periods).",
            "Excessive False Alarms: Hand gesticulation, dropping the device, or clapping triggers spurious alarms.",
            "Physical Boundary Limits: Pressure-sensitive floor mats cover only small doorways; cannot protect full living spaces.",
            "High Installation Costs: Wiring physical sensor matrices under carpets requires intrusive home renovation."
        ],
        card2_title="Vision-Based Contactless Monitoring",
        card2_bullets=[
            "Zero Patient Compliance: Passive overhead or wall-mounted optical monitoring operates 24/7 automatically.",
            "Biomechanical Context: Visual frames confirm whether subject is sitting, bending, lying down, or in distress.",
            "Multi-Person Support: A single optical sensor tracks multiple occupants and preserves individual identities.",
            "Edge Privacy Compliance: Neural pose inference extracts wireframe keypoints; raw video does not leave the room."
        ],
        takeaway="Vision-based pose estimation overcomes the catastrophic compliance failures inherent in wearable medical pendants.",
        card1_color=COLOR_ACCENT_AMBER,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 4: Project Objectives & Core Vision
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 1: Clinical Foundations",
        title="Project Objectives & Research Scope",
        card_title="Core Engineering Goals",
        bullets=[
            "Autonomous Real-Time Fall Detection: Achieve high sensitivity (>96%) and specificity (>98%) on standard consumer CPUs without requiring discrete GPUs.",
            "Continuous Multi-Posture Monitoring: Classify and record continuous daily activities (Standing, Sitting, Walking, Bending, Lying Down) with sedentary analytics.",
            "Interactive 3D Biomechanical Visualization: Develop a custom 3D perspective projection engine rendering an anatomically accurate skeleton with orbit/pan/zoom.",
            "Autonomous Multi-Channel Escalation: Trigger immediate two-phase local acoustic alarms and dispatch push alerts via Telegram Bot and SMTP Email with snapshots.",
            "Open Architecture & Low Cost: Build the entire pipeline on accessible, open-source libraries (PyTorch, OpenCV, MediaPipe, NumPy) for rapid deployment."
        ],
        takeaway="The system bridges deep neural perception, 3D anatomical visualization, and autonomous IoT alerting on commodity hardware.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 5: System Overview & Key Innovations
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 1: Clinical Foundations",
        title="System Overview & Key Innovations",
        card1_title="System Overview",
        card1_bullets=[
            "Unified Processing Pipeline: Seamless 10-stage frame iteration from camera acquisition to 3D rendering and alert dispatch.",
            "Dual-Panel Workstation Interface: Left panel displays annotated video with HUD telemetry; right panel displays interactive 3D skeleton.",
            "Multi-Subject Association: ByteTrack ensures robust multi-occupant state continuity throughout occlusions and floor collisions.",
            "Deterministic State Engine: Finite temporal state machine ensures single-frame anomalies never trigger false emergency alarms."
        ],
        card2_title="Key Technical Innovations",
        card2_bullets=[
            "Extended 35-Keypoint Topology: Synthesizes virtual cervical neck and pelvic sacrum keypoints, creating a true anatomical central spine.",
            "12-Factor Biomechanical Fusion: Merges geometric collapse, velocity spikes, 3D spatial tilt, and post-impact stillness.",
            "Adaptive Pose Scheduling: Halves pose inference load during steady standing states while guaranteeing 100% full-rate detection during falls.",
            "Non-Blocking Asynchronous Alerts: Multithreaded dispatch queue decouples network latency, ensuring 0 FPS drop on the vision stream."
        ],
        takeaway="Four major innovations: Extended 35-keypoint spine, 12-factor fusion, adaptive pose tracking, and non-blocking multi-channel dispatch.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 6: Literature Review & State-of-the-Art
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 2: Literature Review",
        title="Literature Survey & Academic Benchmarking",
        card1_title="Prior Methodological Paradigms",
        card1_bullets=[
            "Bounding Box Thresholding (Noury et al., 2008): Uses height-width aspect ratio; triggers false alarms when tying shoelaces or sitting.",
            "Optical Flow Trajectories (Foroughi et al., 2008): Measures pixel velocity vectors; vulnerable to illumination shifts and pet movements.",
            "Heavy 3D CNNs (SlowFast, I3D, C3D): Exceptional action classification accuracy, but demands 250W+ discrete GPUs, making edge deployment unfeasible.",
            "Multi-Person OpenPose (Cao et al., 2017): Provides bottom-up skeleton maps, but achieves only 4–7 FPS on high-end desktop CPUs."
        ],
        card2_title="Our Proposed Academic Contribution",
        card2_bullets=[
            "Decoupled Perception Hierarchy: Uses YOLOv8 Nano strictly for bounding-box crops, followed by localized MediaPipe top-down pose estimation.",
            "Interpretable Biomechanics: Replaces black-box recurrent networks with explicit kinematic equations (angles, velocities, collapse ratios).",
            "Guaranteed Real-Time Edge Throughput: Delivers 28–30+ FPS sustained throughput on modest dual-core and quad-core CPUs.",
            "Interactive Clinical Visualization: Empowers attending staff to inspect fall kinematics in 3D from any vantage point."
        ],
        takeaway="Our hybrid YOLOv8 + MediaPipe kinematic architecture achieves deep-learning accuracy with lightweight CPU edge efficiency.",
        card1_color=COLOR_ACCENT_MUTED,
        card2_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 7: Biomechanical Principles of Human Falls
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 2: Literature Review",
        title="Biomechanical Kinematics of Human Falls",
        card_title="The Four Sequential Phases of an Accidental Collapse",
        bullets=[
            "Phase 1: Pre-Fall / Initiation: Center-of-Mass (CoM) is perturbed outside the Base-of-Support (BoS). Acceleration begins, accompanied by rapid involuntary arm flailing.",
            "Phase 2: Free-Fall Descent: Subject undergoes downward gravitational acceleration. Vertical hip velocity exceeds 1.2 m/s; torso tilt angle changes by >45 deg in <300 ms.",
            "Phase 3: Impact Shock: Body strikes the ground plane. Bounding box height collapses by >50%, horizontal width expands, and bilateral knees buckle completely (<90 deg flexion).",
            "Phase 4: Post-Impact Resting / Long Lie: Marked by sustained optical landmark stillness. The subject remains prone on the floor plane without recovery movements."
        ],
        takeaway="Accurate detection requires mathematical signatures across ALL 4 phases; isolated thresholding on a single phase generates false positives.",
        accent_color=COLOR_ACCENT_AMBER
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 8: 2D Image Space vs. 3D World Space Dynamics
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 2: Literature Review",
        title="2D Image Space vs. 3D World Space Dynamics",
        card1_title="2D Perspective Ambiguities",
        card1_bullets=[
            "Perspective Foreshortening: A subject falling directly toward or away from the camera lens shows minimal vertical displacement in 2D pixels.",
            "Geometric Confusion: Sitting on a low stool produces identical bounding-box aspect ratio changes (bw/bh) as a prone fall.",
            "Camera Tilt Sensitivity: If the optical axis is tilted 30 deg downward, vertical 2D pixel velocities do not represent true gravity vectors.",
            "Scale Dependency: Pixel velocities vary drastically depending on whether the subject is 2 meters or 6 meters from the camera."
        ],
        card2_title="The 3D World Space Solution",
        card2_bullets=[
            "Metric Coordinates: MediaPipe 3D World Landmarks provide metric (X, Y, Z) coordinates centered at the hips in real-world meters.",
            "True 3D Spatial Torso Tilt: Computes angle relative to gravity vector: theta_3D = arctan(sqrt(dx^2 + dz^2) / |dy|).",
            "Distance Invariance: 3D vector magnitudes remain identical regardless of subject distance or lens focal length.",
            "Anatomical Ground Truth: Enables dynamic tracking of the floor contact plane and realistic 3D skeletal re-projection."
        ],
        takeaway="Extracting 3D spatial world landmarks resolves camera perspective foreshortening that blinds traditional 2D fall detectors.",
        card1_color=COLOR_ACCENT_RED,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 9: High-Level System Architecture
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 3: System Architecture",
        title="High-Level Multi-Tier System Architecture",
        card_title="The Six Functional Architectural Layers",
        bullets=[
            "Layer 1: Input & Ingestion: Captures frames via OpenCV VideoCapture (DirectShow buffer size 1 for webcams, RTSP streams, or offline MP4 files).",
            "Layer 2: Neural Perception: YOLOv8 Nano identifies person bounding boxes; ByteTrack assigns persistent track IDs across frames.",
            "Layer 3: Kinematic Extraction: MediaPipe Pose Landmarker derives 33 landmarks; FeatureExtractor computes 12 biomechanical metrics.",
            "Layer 4: Decision & State Machine: WeightedScorer evaluates probabilistic fall risk; TemporalStateMachine enforces temporal persistence.",
            "Layer 5: Audio & Network Dispatch: AlarmSystem triggers two-phase beeps; AlertDispatcher transmits Telegram photos and SMTP emails.",
            "Layer 6: Workstation Visualization: Dashboard renders 5-tier HUD; SkeletonVisualizer3D renders depth-sorted 3D interactive avatar."
        ],
        takeaway="Strict separation of concerns across 6 modular layers ensures maintainability, deterministic testing, and zero memory leaks.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 10: Complete 10-Stage Processing Sequence
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 3: System Architecture",
        title="Sequential Frame Processing Sequence",
        card1_title="Stages 1 through 5 (Perception)",
        card1_bullets=[
            "1. Camera Acquire: Frame grab from hardware buffer with zero queuing latency.",
            "2. YOLOv8 Detection: Bounding box inference at 480x480 resolution with torch.inference_mode.",
            "3. ByteTrack Tracking: Two-tier IoU + Kalman filter association maintaining subject IDs.",
            "4. MediaPipe Pose: Crop-based landmark estimation with adaptive frame skipping.",
            "5. Feature Extraction: Computes 12 kinematic signals (torso angle, hip velocity, collapse)."
        ],
        card2_title="Stages 6 through 10 (Action & GUI)",
        card2_bullets=[
            "6. Fall Scoring: Multi-factor weighted probabilistic evaluation (0.0 to 1.0).",
            "7. Activity Recognition: Classifies posture (Standing, Sitting, Walking, Bending, Lying).",
            "8. 3D Skeleton Visualizer: Transforms 3D points, depth-sorts, and projects wireframe.",
            "9. Dashboard Render: Renders 5-tier telemetry card, docked tags, and emergency banners.",
            "10. Frame Output & Events: Writes unified MP4 output, checks keyboard/mouse interactions."
        ],
        takeaway="Every frame cycles through all 10 stages in under 33 ms, achieving a continuous 30 FPS monitoring cycle.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 11: Technology Stack & Software Ecosystem
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 3: System Architecture",
        title="Technology Stack & Dependencies",
        card1_title="Computer Vision & Machine Learning",
        card1_bullets=[
            "Python 3.11+: Modern typed asynchronous runtime environment.",
            "Ultralytics YOLOv8 Nano: Anchor-free deep object detection optimized with torch.inference_mode.",
            "Google MediaPipe Tasks: High-throughput BlazePose Landmarker task (pose_landmarker.task).",
            "NumPy & SciPy: Vectorized matrix operations, exponential smoothing, and spatial geometry.",
            "OpenCV 4.8+: Video stream acquisition, GUI windowing, drawing primitives, and video encoding."
        ],
        card2_title="System, Audio & Telemetry",
        card2_bullets=[
            "ByteTrack: Robust multi-object tracking library utilizing dual-threshold Kalman filtering.",
            "Winsound / PyGame: Low-latency kernel acoustic alarm generation with cross-platform fallback.",
            "Requests: REST API communication for Telegram Bot photo and message payloads.",
            "SMTPLib & SSL: Standard library secure TLS/SSL email generation with inline MIME images.",
            "Python-PPTX & Unittest: Native automated verification suite and presentation compilation."
        ],
        takeaway="Built entirely with enterprise-grade, open-source Python frameworks, eliminating recurring proprietary licensing fees.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_AMBER
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 12: Hardware Constraints & CPU-First Design
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 3: System Architecture",
        title="Hardware Constraints & CPU-First Design Philosophy",
        card_title="Overcoming Edge Hardware Bottlenecks",
        bullets=[
            "The Real-World Constraint: Care homes and residential clinics cannot deploy $3,000+ GPU workstations in every bedroom or corridor.",
            "Target Hardware: Designed to run 24/7 on standard consumer edge processors (Intel Core i5/i7, AMD Ryzen 5, or Intel N100 mini-PCs).",
            "Thermal & Power Envelope: Operates within a 25W–45W thermal design envelope without active GPU cooling or noisy server fans.",
            "Algorithmic Efficiency Over Brute Force: Employs crop-level inference, vectorized batch projection, and adaptive scheduling rather than GPU acceleration.",
            "Zero Network Bandwidth Drain: Fully autonomous on-device inference; only sends 15KB text/photo alerts when emergencies occur."
        ],
        takeaway="CPU-first architecture democratizes AI fall detection, enabling cost-effective deployment across hundreds of care-home rooms.",
        accent_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 13: Person Detection with YOLOv8 Nano
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 4: Detection & Tracking",
        title="Person Detection with YOLOv8 Nano",
        card1_title="Architecture & Optimization",
        card1_bullets=[
            "Lightweight Backbone: Modified CSPDarknet53 with C2f cross-stage feature integration.",
            "Decoupled Anchor-Free Head: Separates classification and bounding-box regression tasks for faster convergence.",
            "Inference Mode: Executed under torch.inference_mode() to completely disable autograd computation graphs.",
            "Scaled Resolution: Native 640x640 scaled to 480x480 (imgsz=480), yielding a 43.7% reduction in FLOPS."
        ],
        card2_title="Dynamic Confidence Adaptation",
        card2_bullets=[
            "Normal State Confidence (0.50): High precision threshold during routine activities rejects background false positives.",
            "Fall-Active Confidence (0.25): Automatically lowers threshold during confirmed falls to maintain tracking on flattened bodies.",
            "Post-Fall Recovery (0.35): Hysteresis threshold prevents track loss as the fallen subject begins recovery struggles.",
            "Single-Class Focus: Filters exclusively for Class 0 (Person), skipping all other COCO object classes."
        ],
        takeaway="Dynamic confidence adaptation ensures high-precision detection during walking while preventing track loss when prone.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 14: Multi-Object Tracking with ByteTrack
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 4: Detection & Tracking",
        title="Multi-Object Tracking with ByteTrack",
        card1_title="The Tracking Challenge During Falls",
        card1_bullets=[
            "Posture Distortion: A standing human has height > width; a falling human collapses into width > height, confusing standard trackers.",
            "Motion Blur & Occlusion: Rapid descent generates motion blur, causing detection confidence to drop from 0.85 to 0.35.",
            "SORT Failure Mode: Standard SORT trackers discard low-confidence detections, leading to broken tracks and ID reassignment.",
            "Identity Fragmentation: Reassigning a new ID to a fallen subject loses historical velocity and baseline height data."
        ],
        card2_title="ByteTrack Association Strategy",
        card2_bullets=[
            "First Match Tier: High-confidence boxes (conf >= 0.50) are matched to track hypotheses using Kalman filter predictions and IoU.",
            "Second Match Tier: Unmatched tracks are matched against LOW-confidence boxes (0.25 <= conf < 0.50).",
            "Identity Continuity: Retains the person's original ID (e.g. Person #1) throughout the entire collapse and impact phase.",
            "Kalman State Vector: Tracks [x_center, y_center, aspect_ratio, height, vx, vy, va, vh] for smooth spatial trajectory estimation."
        ],
        takeaway="ByteTrack preserves temporal state identity across the fall impact, preventing fragmented tracking histories.",
        card1_color=COLOR_ACCENT_RED,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 15: Spatial History & Bounding Box Dynamics
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 4: Detection & Tracking",
        title="Spatial History & Bounding Box Dynamics",
        card_title="The Person State Memory Engine",
        bullets=[
            "State Isolation: Every tracked person maintains an isolated dictionary state preserving independent spatial baselines.",
            "Dynamic Baseline Height: baseline_h = (1 - alpha) * baseline_h + alpha * current_h (alpha = 0.05). Adapts slowly to posture without absorbing sudden falls.",
            "Coordinate History Deques: Tracks the last 30 frames of hip midpoints, head coordinates, and aspect ratios in fast O(1) deques.",
            "Velocity Vector Derivation: Computes normalized horizontal walking speed (dx / bh / dt) and vertical descent rates.",
            "Candidate Dwell Counters: Tracks candidate_frames and normal_frames to feed the downstream temporal state machine."
        ],
        takeaway="State isolation allows the system to monitor multi-occupant rooms with zero cross-contamination of kinematic baselines.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 16: MediaPipe Pose Landmarker Architecture
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 5: Pose Estimation",
        title="MediaPipe Pose Landmarker Architecture",
        card1_title="Crop-Level Neural Topology",
        card1_bullets=[
            "Decoupled Detector: We bypass MediaPipe's internal full-frame detector, feeding cropped person boxes directly from YOLOv8.",
            "BlazePose Backbone: Lightweight convolutional architecture with modified bottleneck blocks and skip connections.",
            "Heatmap Regression: Predicts 2D heatmaps for joint locations, followed by a regression head for 3D metric coordinates.",
            "Sub-Millisecond Inference: Cropped patch inference executes in ~7.5 ms on CPU, compared to 35 ms for full-frame estimation."
        ],
        card2_title="Coordinate Output Dualism",
        card2_bullets=[
            "2D Normalized Coordinates: (x, y) in [0.0, 1.0] relative to cropped box, mapped back to frame space for 2D overlay.",
            "3D World Landmarks: Metric (x, y, z) in meters relative to the subject's hip center, independent of camera position.",
            "Visibility & Presence Probabilities: Each keypoint outputs visibility score v in [0.0, 1.0] indicating occlusion level.",
            "Temporal Consistency: Filtered across frames to suppress high-frequency jitter while preserving rapid fall dynamics."
        ],
        takeaway="Feeding YOLOv8 bounding crops directly into MediaPipe eliminates 78% of redundant background neural computations.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 17: The 33 MediaPipe Landmark Topology
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 5: Pose Estimation",
        title="The 33 MediaPipe Landmark Topology",
        card1_title="Anatomical Keypoint Grouping",
        card1_bullets=[
            "Cranial Group (0–10): Nose (0), Eyes (1–6), Ears (7–8), Mouth corners (9–10). Essential for head acceleration and trajectory tracking.",
            "Pectoral & Upper Limbs (11–16): Left/Right Shoulders (11, 12), Elbows (13, 14), Wrists (15, 16). Tracks flailing and defensive bracing.",
            "Distal Hand Extremities (17–22): Pinky (17, 18), Index (19, 20), Thumb (21, 22). Form palm base loops in 3D visualization.",
            "Pelvic Core (23–24): Left/Right Hips. The primary biomechanical anchor for center-of-mass and vertical descent calculation."
        ],
        card2_title="Lower Extremities & Base Support",
        card2_bullets=[
            "Lower Articulations (25–28): Left/Right Knees (25, 26), Left/Right Ankles (27, 28). Governs knee collapse and posture classification.",
            "Base Support Extremities (29–32): Left/Right Heels (29, 30), Left/Right Foot Index / Toes (31, 32).",
            "Ground Contact Tracking: Lowest foot keypoint (y_min) dynamically establishes the real-world floor plane contact.",
            "Standard Pose Connections: 35 anatomical wireframe lines connect articulations into a cohesive human skeleton."
        ],
        takeaway="The 33 landmarks capture all primary biomechanical degrees of freedom required to model human collapse kinematics.",
        card1_color=COLOR_ACCENT_MUTED,
        card2_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 18: Landmark Confidence & Occlusion Handling
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 5: Pose Estimation",
        title="Landmark Confidence Scoring & Occlusion Handling",
        card_title="Robust Fallbacks for Partial Body Occlusion",
        bullets=[
            "Visibility Gate: Connections are drawn only if min(vis_a, vis_b) >= 0.30, preventing erratic 'ghost' limbs when occluded by tables or beds.",
            "Virtual Hip Substitution: If hip landmarks are occluded, the system computes the centroid from the lower 30% of the YOLO bounding box.",
            "Knee Occlusion Fallback: When feet are obscured, knee collapse and aspect-ratio indicators carry increased weight in the scoring engine.",
            "Occlusion Grace Period: The system tolerates up to 30 frames of missing pose landmarks during an active fall before declaring track loss.",
            "Adaptive Wireframe Ghosting: Occluded joints in the 3D visualizer are rendered with 50% opacity rather than abruptly disappearing."
        ],
        takeaway="Multi-tier confidence fallbacks ensure uninterrupted safety monitoring even when furniture partially obscures the body.",
        accent_color=COLOR_ACCENT_AMBER
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 19: The 12 Biomechanical Signals Overview
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 6: Biomechanical Engine",
        title="The 12 Biomechanical Signals Overview",
        card1_title="Geometric & Velocity Signals (1–6)",
        card1_bullets=[
            "1. Torso Angle (deg): Angular deviation of spine vector from vertical axis (0 deg = standing, 90 deg = horizontal).",
            "2. Aspect Ratio (AR): Ratio of bounding box width to height (bw / bh). Normal: 0.3; Fall: >1.2.",
            "3. Height Collapse Ratio: Normalized drop in bounding box height relative to baseline: max(0, 1 - bh / baseline_h).",
            "4. Width Expansion Ratio: Normalized horizontal spread: max(0, bw / baseline_w - 1).",
            "5. Vertical Hip Velocity: Downward velocity of hip midpoint: dy_hip / (bh * dt).",
            "6. Upward Hip Velocity: Upward velocity indicating recovery or standing up: max(0, -dy_hip / (bh * dt))."
        ],
        card2_title="Structural & Kinetic Signals (7–12)",
        card2_bullets=[
            "7. Head Velocity: Vertical descent rate of the nose keypoint; captures rapid whipping during trips.",
            "8. Shoulder-Hip Separation: Vertical distance between shoulders and hips; collapses during spinal flexion.",
            "9. Bilateral Knee Collapse: Minimum knee angle; detects involuntary leg buckling (<90 deg).",
            "10. 3D Torso Tilt: Metric 3D spine angle relative to gravity; resolves 2D foreshortening ambiguities.",
            "11. Bilateral Asymmetry: Asymmetric leg extension indicating uncontrolled side or spiral falls.",
            "12. Post-Impact Stillness: Motionlessness metric computed across 15 frames post-collapse."
        ],
        takeaway="Fusing 12 complementary biomechanical signals eliminates single-point-of-failure false alarms.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 20: Primary Geometric Collapse Features
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 6: Biomechanical Engine",
        title="Primary Geometric Collapse Formulation",
        card1_title="Aspect Ratio & Height Collapse Math",
        card1_bullets=[
            "Bounding Box Aspect Ratio: AR = (x2 - x1) / (y2 - y1) = bw / bh.",
            "Upright Standing Profile: AR in [0.25, 0.45]; Fallen Prone Profile: AR in [1.2, 3.2].",
            "Height Collapse Equation: H_collapse = max(0.0, 1.0 - (bh / baseline_h)).",
            "Adaptive Baseline Update: baseline_h = (1 - alpha)*baseline_h + alpha*bh (alpha=0.05).",
            "Anti-Absorption Guard: Baseline adaptation freezes during candidate falls to prevent absorbing collapse."
        ],
        card2_title="Width Expansion & Torso Angle",
        card2_bullets=[
            "Width Expansion Equation: W_expand = max(0.0, (bw / baseline_w) - 1.0).",
            "Lateral Ground Sprawl: Captures limbs spreading across floor during impact.",
            "2D Torso Angle Formulation: theta_2D = abs(degrees(arctan2(dx_spine, dy_spine))).",
            "Spine Vector: dx = (ls.x + rs.x)/2 - (lh.x + rh.x)/2; dy = (ls.y + rs.y)/2 - (lh.y + rh.y)/2.",
            "Angle Score Clamp: s_angle = clamp01((theta_2D - 30.0) / 45.0)."
        ],
        takeaway="Freezing baseline updates during candidate falls preserves the reference height needed to detect collapse.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_AMBER
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 21: Kinematic Velocity & Acceleration Features
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 6: Biomechanical Engine",
        title="Kinematic Velocity & Acceleration Modeling",
        card_title="Scale-Invariant Dynamic Derivatives",
        bullets=[
            "Vertical Hip Descent Velocity: v_hip = (y_hip(t) - y_hip(t - dt)) / (bh * dt). Normalizing by bounding box height (bh) makes velocity scale-invariant across distances.",
            "Head Whipping Velocity: v_head = (y_nose(t) - y_nose(t - dt)) / (bh * dt). In forward and backward trips, rotational torque causes the head to descend faster than the hips.",
            "Velocity Clamping & Scoring: s_vel = clamp01((v_hip - 0.08) / 0.35). Rejects slow, controlled sitting motions (v < 0.08 bh/s) while triggering on rapid falls.",
            "Angular Velocity Delta: ang_vel = abs(theta(t) - theta(t - dt)) / dt. Captures rapid angular pitching exceeding 60 deg/sec.",
            "Directional Hip Vector: Distinguishes downward collapse (+dy) from intentional jumping or standing recovery (-dy)."
        ],
        takeaway="Height-normalized velocity equations ensure a person 6 meters away triggers identical velocity scores as a person 2 meters away.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 22: Advanced Structural & Impact Features
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 6: Biomechanical Engine",
        title="Advanced Structural & Impact Kinematics",
        card1_title="Bilateral Knee Collapse Angle",
        card1_bullets=[
            "Flexion Vector Math: kappa = arccos( (u . v) / (||u|| * ||v||) ), where u = Hip - Knee, v = Ankle - Knee.",
            "Waist Bending vs. Leg Buckling: Tying shoes maintains straight knees (kappa > 145 deg); falls induce rapid knee collapse (kappa < 85 deg).",
            "Bilateral Min Evaluation: kappa_min = min(kappa_left, kappa_right).",
            "Knee Collapse Score: s_knee = clamp01((120.0 - kappa_min) / 60.0)."
        ],
        card2_title="3D Tilt & Post-Impact Stillness",
        card2_bullets=[
            "3D Spatial Torso Tilt: theta_3D = degrees(arctan2(sqrt(dx^2 + dz^2), |dy|)). Invariant to optical camera angle.",
            "Stillness Motion Metric: M_still = sum(||lm(t) - lm(t-1)||) / (bh * N_landmarks).",
            "Post-Impact Immobility: Evaluated across 15 frames following velocity spike; confirms incapacitation.",
            "Stillness Score: s_still = clamp01(1.0 - (M_still / 0.02)). High score indicates absence of recovery motion."
        ],
        takeaway="Knee collapse and post-impact stillness reliably differentiate intentional floor exercises from accidental falls.",
        card1_color=COLOR_ACCENT_GREEN,
        card2_color=COLOR_ACCENT_AMBER
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 23: Multi-Factor Weighted Scoring Engine
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 7: Scoring & State Machine",
        title="Multi-Factor Weighted Scoring Formulation",
        card_title="Probabilistic Kinematic Fusion",
        bullets=[
            "Sigmoidal Normalization: Every kinematic feature is mapped to a normalized sub-score s_i in [0.0, 1.0] using piecewise linear clamps.",
            "Composite Fall Probability Score: S_fall = sum_{i=1}^{12} w_i * s_i, where sum w_i = 1.0.",
            "Deterministic Interpretability: Every sub-score s_i is logged and rendered on the HUD factor panel, enabling clinicians to audit why a fall triggered.",
            "Score Thresholds: S_fall >= 0.55 triggers immediate confirmed fall; S_fall in [0.40, 0.54] initiates temporal candidate accumulation.",
            "Persistent Score History: Running average over 10 frames prevents single-frame landmark dropouts from prematurely resetting alarms."
        ],
        takeaway="Weighted multi-factor scoring provides a transparent, auditable probability score with zero black-box opacity.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 24: Signal Weight Distribution & Sensitivity
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 7: Scoring & State Machine",
        title="Signal Weight Distribution & Calibration",
        card1_title="Primary Signal Weights",
        card1_bullets=[
            "Torso Angle Weight (0.22): Strongest single indicator of horizontal body posture.",
            "Vertical Hip Velocity (0.18): Essential prerequisite for true dynamic impact.",
            "Height Collapse Ratio (0.15): Captures major vertical volume reduction.",
            "Head Whipping Velocity (0.10): Detects high-momentum forward/backward trips.",
            "Bounding Box Aspect Ratio (0.09): Confirms prone, floor-bound orientation.",
            "3D Spatial Torso Tilt (0.08): Perspective-invariant orientation check."
        ],
        card2_title="Secondary Confirmatory Weights",
        card2_bullets=[
            "Post-Impact Stillness (0.08): Confirms inability to stand back up.",
            "Knee Collapse Angle (0.06): Differentiates bending from leg buckling.",
            "Shoulder-Hip Separation (0.04): Verifies vertical spine compression.",
            "Width Expansion Ratio (0.04): Detects floor sprawl.",
            "Bilateral Asymmetry (0.02): Captures lateral rotational falls.",
            "Angular Velocity Delta (0.02): Verifies rapid rotational transition."
        ],
        takeaway="Calibrated empirically across Le2i and simulated clinical datasets to optimize the sensitivity/specificity trade-off.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 25: Temporal State Machine (FSM) Architecture
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 7: Scoring & State Machine",
        title="Temporal Finite State Machine (FSM)",
        card1_title="State Definitions",
        card1_bullets=[
            "NORMAL State: Subject upright or engaged in controlled routine movement; counters clear.",
            "CANDIDATE State: S_fall >= 0.40; increments candidate_frames counter on every tick.",
            "CONFIRMED (FALLEN) State: candidate_frames >= 8 (or S_fall >= 0.55). Triggers alarm and alert dispatcher.",
            "RECOVERING State: Subject uprighting; monitored for rapid stabilization."
        ],
        card2_title="Temporal Hysteresis Logic",
        card2_bullets=[
            "8-Frame Confirmation Buffer: Requires ~270 ms of sustained collapse evidence to reject momentary camera glitches.",
            "20-Frame Standard Reset: Requires ~700 ms of continuous normal scores to clear an unconfirmed candidate.",
            "Instant High-Score Bypass: S_fall >= 0.55 immediately promotes candidate to CONFIRMED without waiting for 8 frames.",
            "Floor Timer Integration: Stamps fall_start_time and accumulates continuous fall_duration."
        ],
        takeaway="The temporal state machine acts as a temporal low-pass filter, eliminating transient single-frame false triggers.",
        card1_color=COLOR_ACCENT_AMBER,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 26: State Transitions & Geometric Recovery Reset
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 7: Scoring & State Machine",
        title="State Transitions & Geometric Recovery Reset",
        card_title="Autonomous Recovery Detection",
        bullets=[
            "Standard Reset Path: If normal_frames >= 20, the FSM transitions from FALLEN back to NORMAL.",
            "Fast Geometric Recovery Reset: If (AR < 0.35) and (theta_torso < 18 deg) and (normal_frames >= 5): Immediately resets without waiting for 20 frames.",
            "Clinical Rationale: When an elderly person clearly regains an upright standing posture, holding the alarm active for 20 frames causes unnecessary panic.",
            "Recovery Alert Trigger: Reset event captures total floor duration and dispatches a 'Subject Recovered' push notification to caregivers.",
            "Alarm Silence: Disables Phase 2 audio siren instantly upon confirmed geometric recovery."
        ],
        takeaway="Fast geometric reset silences alarms the exact moment a patient stands up, recording precise floor dwell times.",
        accent_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 27: Multi-Posture Activity Monitoring
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 8: Activity Monitoring",
        title="Multi-Posture Activity Classification",
        card1_title="Classification Heuristics",
        card1_bullets=[
            "STANDING: AR < 0.50, Torso theta < 25 deg, Knee kappa > 155 deg, Speed < 0.08 bh/s.",
            "WALKING: AR < 0.55, Torso upright, Horizontal hip velocity vx >= 0.08 bh/s.",
            "SITTING: AR in [0.55, 0.95], Torso upright (theta < 35 deg), Knee flexion kappa in [70, 120 deg].",
            "BENDING: Torso theta in [40, 75 deg], Knees straight (kappa > 140 deg), Drop < 0.25.",
            "LYING DOWN: Horizontal AR > 1.10, Torso theta > 70 deg, Absence of descent velocity spike."
        ],
        card2_title="Concurrent Execution Architecture",
        card2_bullets=[
            "Shared Feature Graph: Reuses the exact same 12-factor feature vector extracted for fall detection.",
            "Zero Overhead: Activity classification consumes <0.01 ms per frame, running concurrently with fall scoring.",
            "HUD Tag Integration: Activity labels dock directly above the person's bounding box: [P1 | STANDING (98%)].",
            "Clinical Telemetry: Distinguishes between accidental falls and controlled intentional lying on beds/couches."
        ],
        takeaway="Concurrent activity classification provides continuous operational context without additional neural network passes.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 28: Debounce Filtering & Smoothing
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 8: Activity Monitoring",
        title="Debounce Filtering & Temporal Smoothing",
        card1_title="The Posture Flickering Problem",
        card1_bullets=[
            "Natural Articulation Variance: Shifting weight while standing or taking a step causes momentary knee angle fluctuation.",
            "Unfiltered Artifacts: Raw frame-by-frame classification causes HUD labels to flicker between Standing and Walking.",
            "User Experience Impact: Rapid flickering degrades visual readability and distorts cumulative duration logs."
        ],
        card2_title="Debounce & Majority Consensus",
        card2_bullets=[
            "FIFO Circular Buffer: Stores the last 7 raw classified activity states in a rolling queue.",
            "Consensus Voting: Smoothed activity requires a >=60% majority consensus across the window.",
            "State Dwell Persistence: A new posture must persist for at least 5 frames before transitioning the HUD display.",
            "Smooth Duration Tracking: Prevents split-second transitions from fragmenting the healthcare activity ledger."
        ],
        takeaway="Consensus debounce filtering eliminates label flickering, ensuring clean telemetry and accurate healthcare logs.",
        card1_color=COLOR_ACCENT_AMBER,
        card2_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 29: Posture Duration Ledger & Sedentary Telemetry
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 8: Activity Monitoring",
        title="Posture Duration Ledger & Sedentary Analytics",
        card_title="Continuous Healthcare Telemetry",
        bullets=[
            "Cumulative Time Accounting: Tracks cumulative time in seconds for every posture: T_sit, T_stand, T_walk, T_lie.",
            "Dynamic Percentage Ledger: %_sitting = (T_sitting / T_total) * 100. Displayed live on the 5-tier HUD card.",
            "Sedentary Behavior Warning: If T_total >= 10 minutes and (%_sitting + %_lying) > 65%, flags a SEDENTARY warning.",
            "Mobility Trend Analytics: Long-term drift toward higher sitting percentages indicates physical decline or fall risk.",
            "Structured CSV Archiving: Writes cumulative activity summaries to session logs for physician review."
        ],
        takeaway="Transforms an emergency fall detector into an all-day preventative wellness and mobility monitoring platform.",
        accent_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 30: Motivation for Custom 3D Visualization
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 9: 3D Skeleton Engine",
        title="Motivation for Custom 3D Projection Engine",
        card1_title="Limitations of External Viewers",
        card1_bullets=[
            "OpenGL / Open3D Context Switching: Running separate GLFW/OpenGL threads causes GPU context switching and thread locks.",
            "Recording Impossibility: Capturing dual-panel MP4 video with external OpenGL windows requires slow desktop screen-grabbing.",
            "Heavyweight Footprint: External 3D engines add 200MB+ in dependencies and introduce cross-platform driver crashes.",
            "UI Disconnect: External windows cannot be cleanly docked side-by-side inside a single OpenCV window."
        ],
        card2_title="Our Custom OpenCV Engine",
        card2_bullets=[
            "Pure NumPy & OpenCV: 100% vectorized 3D-to-2D matrix projection engine requiring zero external 3D libraries.",
            "Unified Horizontal Canvas: Concatenates frame and 3D canvas into a single output: np.hstack([frame, canvas_3d]).",
            "Single Video Stream: Seamlessly encoded into output MP4 files at full 30 FPS by a single cv2.VideoWriter instance.",
            "Low Latency: Executes depth-sorted 3D projection and rendering in only 2.17 ms per frame."
        ],
        takeaway="A custom vectorized 3D projection engine enables side-by-side integration with zero external window dependencies.",
        card1_color=COLOR_ACCENT_MUTED,
        card2_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 31: 3D Camera Geometry & Perspective Math
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 9: 3D Skeleton Engine",
        title="3D Camera Geometry & Perspective Mathematics",
        card_title="Trigonometric World-to-Screen Mapping",
        bullets=[
            "Metric World Space Translation: P_shifted = P_world - C, where C = (cx, cy, cz) is the virtual camera focal anchor.",
            "Euler Rotation Matrix: R = Rx(elevation) * Ry(azimuth). Azimuth orbits horizontally (0-360 deg); Elevation tilts vertically (-89 to +89 deg).",
            "Camera Space Coordinates: P_cam = P_shifted * R^T, yielding rotated 3D coordinates (X_cam, Y_cam, Z_cam).",
            "Perspective Division: u = (f * X_cam) / (Z_cam + d_cam) + (W / 2); v = -(f * Y_cam) / (Z_cam + d_cam) + (H / 2).",
            "Vectorized Batch Projection: Projects all 35 keypoints simultaneously in a single NumPy matrix multiplication pass."
        ],
        takeaway="Vectorized matrix projection projects the full 35-keypoint skeleton in under 0.15 ms on CPU.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 32: Extended Anatomical Topology (33 -> 35 Keypoints)
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 9: 3D Skeleton Engine",
        title="Extended Anatomical Topology (33 -> 35 Points)",
        card1_title="MediaPipe Topological Flaws",
        card1_bullets=[
            "Missing Central Spine: MediaPipe connects shoulders directly to hips (11->23, 12->24), creating an empty, hollow torso.",
            "Disconnected Head: Cranial keypoints float above shoulders without a neck or cervical spine connection.",
            "Pelvic Void: Hip joints (23, 24) float disconnected from the lumbar spine.",
            "Open Extremities: Hands and feet terminate as loose lines without anatomical base loops."
        ],
        card2_title="Synthesized Virtual Keypoints",
        card2_bullets=[
            "Virtual Keypoint 33 (Neck): P_33 = 0.5 * (P_11 + P_12). Midpoint of shoulders.",
            "Virtual Keypoint 34 (Pelvis): P_34 = 0.5 * (P_23 + P_24). Midpoint of hips.",
            "True Central Spine: Nose (0) -> Neck (33) -> Mid-Torso -> Pelvis (34).",
            "Clavicular Girdle & Pelvic Basin: Connects shoulders to neck, and hips to pelvis.",
            "Closed Extremity Loops: Palm triangles (Wrist-Index-Pinky) and foot arches (Ankle-Heel-Toe)."
        ],
        takeaway="Synthesizing virtual neck and pelvis keypoints transforms a hollow wireframe into an anatomically authentic human skeleton.",
        card1_color=COLOR_ACCENT_RED,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 33: Multi-Tier Visual Hierarchy & Depth Sorting
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 9: 3D Skeleton Engine",
        title="Multi-Tier Visual Hierarchy & Depth Sorting",
        card1_title="Depth-Sorted Painter's Algorithm",
        card1_bullets=[
            "Camera Depth Assignment: Every joint and bone is tagged with its transformed Z_cam depth value.",
            "Descending Depth Sort: Elements are sorted from farthest (Z_max) to nearest (Z_min).",
            "Correct Occlusion Rendering: Distant limbs are rendered first; foreground limbs draw on top, preventing visual inversion.",
            "Depth-Scaled Bone Thickness: thickness = clamp(round(base_thick * (d_ref / Z_cam)), 1, 6)."
        ],
        card2_title="3-Tier Joint Visual Hierarchy",
        card2_bullets=[
            "Tier 1 (Root Structural Anchors): Pelvis, Neck, Hips, Shoulders (1.30x radius scale).",
            "Tier 2 (Major Articulations): Knees, Elbows, Ankles, Wrists (1.05x radius scale).",
            "Tier 3 (Peripheral Extremities): Hands, Feet, Facial Anchors (0.75x radius scale).",
            "Anti-Aliased Halos & Specular Cores: Every joint features a dark anti-aliased halo, colored disc, and offset specular highlight."
        ],
        takeaway="Depth sorting and multi-tier joint scaling provide genuine 3D visual depth without requiring a full 3D graphics pipeline.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 34: Adaptive Centering, Framing & Ground Grid
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 9: 3D Skeleton Engine",
        title="Adaptive Centering, Camera Framing & Ground Grid",
        card_title="Intelligent Viewport Kinematics",
        bullets=[
            "Adaptive 55%–70% Viewport Scaling: Dynamically computes camera distance: d_cam = (f * s_ref) / (coverage * H). Human figure automatically occupies 62% of viewport height regardless of posture.",
            "Stable Pelvis Midpoint Centering: Camera focal target locks to the hip midpoint P_34 = (P_23 + P_24) / 2 with a +0.07m vertical visual bias.",
            "EMA Jitter Filtering: Camera center and zoom interpolate with alpha_pos = 0.15 and alpha_dist = 0.10, eliminating landmark micro-jitter.",
            "Dynamic Floor Tracking: y_floor = min(y_ankles, y_heels, y_toes). Ground grid anchors beneath the lowest foot contact point.",
            "Interactive Mouse Control: Left-Drag orbits azimuth/elevation; Right-Drag / Wheel zooms; Double-Click resets to default framing."
        ],
        takeaway="Adaptive bounding-box scaling and hip centering ensure the subject is always framed perfectly, whether standing or collapsed.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 35: The Performance Discrepancy & Profiler Design
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 10: Performance Optimization",
        title="Performance Discrepancy & Profiler Design",
        card1_title="The Initial 8.9 FPS Bottleneck",
        card1_bullets=[
            "The Discrepancy: Baseline testing achieved only ~8.9 FPS, failing clinical real-time requirements (>=25 FPS).",
            "Excessive Frame Latency: Total frame processing took 112.4 ms per frame, causing video lag and dropped frames.",
            "Diagnostic Need: Needed an exact, nanosecond-resolution breakdown to identify the true pipeline bottleneck.",
            "Suspicion vs. Reality: Initial assumption blamed 3D rendering; profiling revealed YOLOv8 and MediaPipe consumed 92% of runtime."
        ],
        card2_title="10-Stage Pipeline Profiler",
        card2_bullets=[
            "High-Resolution Clock: Uses time.perf_counter() around each of the 10 processing stages.",
            "Rolling Window Statistics: Computes rolling 30-frame means and percentage distributions.",
            "Zero Profiling Overhead: Latency tracking consumes <0.05 ms per frame.",
            "Instant Diagnostics: Prints formatted summary table upon session exit or CLI request."
        ],
        takeaway="High-resolution profiling revealed that perception models, not 3D rendering, were the primary source of latency.",
        card1_color=COLOR_ACCENT_RED,
        card2_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 36: Key Acceleration Techniques Implemented
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 10: Performance Optimization",
        title="Key Acceleration Techniques Implemented",
        card_title="Algorithmic CPU Optimizations",
        bullets=[
            "PyTorch Inference Mode: Wrapped YOLOv8 calls with @torch.inference_mode(), disabling autograd tracking and reducing memory allocations.",
            "YOLOv8 Resolution Scaling: Reduced inference size from 640x640 to 480x480 (imgsz=480), cutting convolution FLOPS by 43.7%.",
            "Adaptive Pose Estimation: During steady NORMAL states with bounding-box delta < 8 px, MediaPipe pose alternates frames. During candidate or fall states, runs at 100% full frame rate.",
            "Vectorized NumPy 3D Math: Replaced Python loops with batch matrix multiplications: P_cam = P_world * R^T.",
            "Fast In-Place ROI Blending: Replaced full-frame np.copy() with localized ROI cv2.addWeighted, cutting UI drawing from 4.5 ms to 0.47 ms."
        ],
        takeaway="Combined optimizations reduced frame processing latency from 112 ms to 32 ms, increasing frame rate from 8.9 to 30.5 FPS.",
        accent_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 37: Benchmark Profiling Results Across Stages
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 10: Performance Optimization",
        title="Benchmark Profiling Results Across Stages",
        card1_title="Post-Optimization Stage Latency",
        card1_bullets=[
            "camera_acquire: 1.82 ms (0.6% total time)",
            "yolo_detection: 22.40 ms (68.2% total time)",
            "bytetrack_tracking: 0.01 ms (<0.1% total time)",
            "mediapipe_pose: 7.50 ms (22.8% total time)",
            "feature_extraction: 0.01 ms (<0.1% total time)",
            "fall_scoring & activity: 0.01 ms (<0.1% total time)",
            "visualizer_3d: 2.17 ms (6.6% total time)",
            "dashboard_render: 0.47 ms (1.4% total time)",
            "frame_output: 0.80 ms (2.4% total time)"
        ],
        card2_title="Performance Summary Metrics",
        card2_bullets=[
            "Total Frame Latency: 32.8 ms (Down from 112.4 ms baseline).",
            "Sustained Frame Rate: 30.5 FPS on standard Intel Core CPU.",
            "3D Viewport Latency: Only 2.17 ms for full depth-sorted projection.",
            "Dashboard HUD Latency: Only 0.47 ms using in-place ROI blending.",
            "CPU Utilization: Stable 42% on quad-core processor without thermal throttling."
        ],
        takeaway="The optimized pipeline achieves sustained 30+ FPS throughput on edge CPUs, meeting all clinical real-time requirements.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 38: Dual-Panel Workstation Layout Architecture
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 11: Workstation UI/UX",
        title="Dual-Panel Workstation Layout Architecture",
        card1_title="Left Panel: Native Camera Stream",
        card1_bullets=[
            "58% Display Width: Live camera feed with original aspect ratio preservation.",
            "Docked Bounding Box Tags: Displays person ID, current activity, and confidence above box.",
            "2D Pose Wireframe: Overlays 2D MediaPipe skeleton with color-coded joint confidence.",
            "Top-Left HUD Card: 5-tier telemetry card displaying real-time system state.",
            "Bottom Emergency Banners: Displays high-contrast alert notifications during incidents."
        ],
        card2_title="Right Panel: 3D Skeleton Viewport",
        card2_bullets=[
            "42% Display Width: Dedicated interactive 3D virtual environment.",
            "Vertical Divider Border: 2px dividing border (#2A2634 / #444252) clearly separates panels.",
            "3D Status Header: Displays virtual camera orientation (Azimuth, Elevation, Zoom).",
            "Perspective Ground Grid: Anchored to subject's foot contact plane.",
            "Mouse Interaction: Direct orbit, pan, and zoom controls mapped via x_offset."
        ],
        takeaway="The dual-panel architecture provides simultaneous real-world video context and abstract 3D biomechanical analysis.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 39: 5-Tier Telemetry Information Hierarchy
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 11: Workstation UI/UX",
        title="5-Tier Telemetry Information Hierarchy",
        card_title="Structured HUD Telemetry Card (alpha = 0.22, card_w = 360px)",
        bullets=[
            "Header Bar: System Title ('AI SAFETY MONITOR') + Prominent Status Pill (NORMAL, WARNING, FALL DETECTED, RECOVERY).",
            "Tier 1 (Activity): ACTIVITY: STANDING — 98% [P1]. Color-coded by classified posture.",
            "Tier 2 (Tracking): TRACKING: 2 Detected [P1, P2]. Displays active occupant count and unique track IDs.",
            "Tier 3 (Incidents): INCIDENTS: 1 Confirmed (Last: 14:22:05). Historical event count and timestamp.",
            "Tier 4 (Severity): SEVERITY: Score: 0.88 | Level: SEVERE. Real-time kinematic score and severity.",
            "Tier 5 (Performance): PERF: 29.8 FPS (avg: 29.1) | 33.4 ms. High-resolution engine latency from profiler.",
            "Footer (Postures): POSTURES: Sit: 5% | Stand: 85% | Walk: 10%. Cumulative session breakdown."
        ],
        takeaway="Structured 5-tier hierarchy allows attending nurses to assess safety status in under 2 seconds.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 40: Docked Bounding Box Tags & Emergency Banners
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 11: Workstation UI/UX",
        title="Docked Bounding Box Tags & Emergency Banners",
        card1_title="Docked Bounding Box Header Tags",
        card1_bullets=[
            "Eliminating Clutter: Replaced floating raw text with clean docked header pills.",
            "Docked Placement: Anchored directly above top edge of box; flips inside if near screen top.",
            "Information Density: Displays P1 | STANDING — 98% or P1 | FALLEN [SEVERE] 4.2s.",
            "Corner Accent Brackets: Clean corner brackets highlight target without obscuring body."
        ],
        card2_title="High-Visibility Emergency Banners",
        card2_bullets=[
            "Confirmed Fall Banner: Red translucent card at screen bottom: ! CRITICAL: FALL DETECTED — P1 | SEVERE | 3.2s | Score: 0.89 !",
            "Recovery Banner: Cerulean blue translucent banner: i RECOVERY DETECTED — P1 STABILIZING.",
            "Track Loss Warning: ! CRITICAL ALERT: PERSON DOWN ON GROUND — TRACK LOST !.",
            "Clamped Factor Panel: 12-factor bar panel strictly clamped to frame boundaries, preventing screen overflow."
        ],
        takeaway="Docked tags and high-contrast emergency banners keep the video feed clean while highlighting critical events.",
        card1_color=COLOR_ACCENT_AMBER,
        card2_color=COLOR_ACCENT_RED
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 41: Two-Phase Audio Alarm Escalation System
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 12: Alerts & Escalation",
        title="Two-Phase Audio Alarm Escalation System",
        card1_title="Two-Phase Acoustic Escalation",
        card1_bullets=[
            "Phase 1 (Initial Alert): Soft double-beep (700 Hz, 250 ms duration, 120 ms gap). Triggered immediately upon confirmed fall.",
            "Clinical Purpose: Notifies nearby caregivers or the resident without causing acoustic panic.",
            "Phase 2 (Critical Escalation): Continuous rapid high-pitch beeps (1400 Hz, 120 ms pulse) if subject remains on floor >= 5.0 seconds.",
            "High-Risk Signal: Warns staff that patient is immobilized, injured, or unable to self-recover."
        ],
        card2_title="Autonomous Acoustic Reset & Backend",
        card2_bullets=[
            "Autonomous Silence: Alarm terminates immediately when the temporal state machine confirms upright recovery.",
            "Native Windows Winsound: Utilizes kernel-level winsound.Beep for zero-latency acoustic generation.",
            "Cross-Platform Fallback: Automatically falls back to PyGame mixer audio on Linux or macOS.",
            "Independent Thread: Audio synthesis executes on an independent background thread without blocking video processing."
        ],
        takeaway="Two-phase escalation differentiates momentary stumbles from severe, immobilized long-lie emergencies.",
        card1_color=COLOR_ACCENT_AMBER,
        card2_color=COLOR_ACCENT_RED
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 42: Instant Telegram Bot Alert Dispatcher
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 12: Alerts & Escalation",
        title="Instant Telegram Bot Alert Dispatcher",
        card1_title="Direct Push Alert Architecture",
        card1_bullets=[
            "Telegram Bot API Integration: Communicates directly with Telegram cloud servers via HTTPS POST requests.",
            "Dedicated Push Channel: Sends alerts instantly to caregivers' smartphones or central nurse station channels.",
            "Markdown Formatted Telemetry: Subject ID, Fall Severity, Confidence Score, Floor Duration, Prior Activity, and Timestamp.",
            "Recovery Confirmation: Dispatches a follow-up notification when subject stands up: STATUS UPDATE: SUBJECT RECOVERED."
        ],
        card2_title="Automated Image Attachment",
        card2_bullets=[
            "Annotated Incident Snapshot: Captured frame is watermarked with an emergency red banner and saved to disk.",
            "sendPhoto API Call: Transmits the annotated JPEG snapshot directly into the Telegram chat.",
            "Instant Visual Triage: Allows clinicians to remotely verify whether a fall is severe or a false alarm.",
            "Live Verified: Successfully tested with real bot tokens and chat IDs with zero delivery failures."
        ],
        takeaway="Transmitting an annotated photo directly to caregivers' phones allows instant remote medical triage.",
        card1_color=COLOR_ACCENT_CYAN,
        card2_color=COLOR_ACCENT_GREEN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 43: SMTP Email Dispatch & Asynchronous Thread Pool
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 12: Alerts & Escalation",
        title="SMTP Email Dispatch & Asynchronous Thread Pool",
        card_title="Non-Blocking Enterprise Messaging Architecture",
        bullets=[
            "Multipart MIME HTML Email: Builds rich HTML emails with emergency tables, severity tags, and embedded inline snapshots (cid:snapshot_img).",
            "Secure SMTP Handshake: Connects via TLS (Port 587) or SSL (Port 465) with Google App Passwords or corporate mail servers.",
            "Asynchronous ThreadPoolExecutor: Network HTTP and SMTP calls (500–2000 ms latency) run inside a background ThreadPoolExecutor(max_workers=2).",
            "Zero FPS Drop: Enqueuing alerts takes <0.5 ms on the main thread, guaranteeing zero video lag or dropped frames.",
            "Smart 30-Second Cooldown: Suppresses duplicate alert dispatching while a subject remains on the floor; independent timers per person ID."
        ],
        takeaway="Asynchronous thread-pool dispatch decouples network latency, ensuring alerts never degrade video frame rates.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 44: Comprehensive 60-Test Automated Validation Suite
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 13: Verification & Results",
        title="Comprehensive 60-Test Automated Validation",
        card1_title="Test Suite Breakdown (60 Tests)",
        card1_bullets=[
            "test_dashboard_hierarchy.py (11 tests): Validates all 10 operational states, HUD hierarchy, and 3D boundary line.",
            "test_alert_dispatcher.py (7 tests): Validates async execution, simulation mode, cooldown suppression, and recovery dispatch.",
            "test_edge_cases.py (12 tests): Validates sub-ms clock ticks (dt=0), camera FPS anomalies (NaN/0.0), and 5000 FPS extremes.",
            "test_framing_and_centering.py (10 tests): Validates 55%–70% adaptive scaling and hip midpoint centering.",
            "test_visual_validation.py (6 tests): Validates depth-sorted 3D projection across 6 standard postures.",
            "test_activity_monitor.py & integration (14 tests): Validates FSM transitions, ledger tracking, and consensus debounce."
        ],
        card2_title="Test Execution Benchmark",
        card2_bullets=[
            "Total Unit Tests: 60 automated tests.",
            "Test Execution Time: Ran 60 tests in 5.200s.",
            "Pass Rate: 100% (60/60 OK, 0 failures, 0 errors).",
            "Continuous Integration: All core modules validated against synthetic frames, video streams, and simulated falls.",
            "Edge Case Resilience: Proved complete mathematical immunity to division by zero and corrupt hardware queries."
        ],
        takeaway="100% pass rate across 60 automated tests guarantees academic and production stability under all edge conditions.",
        card1_color=COLOR_ACCENT_GREEN,
        card2_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 45: Experimental Accuracy & Confusion Matrix
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 13: Verification & Results",
        title="Experimental Accuracy & Confusion Matrix",
        card1_title="Quantitative Accuracy Benchmarks",
        card1_bullets=[
            "Evaluation Dataset: Evaluated across Le2i Fall Detection benchmarks and simulated indoor living datasets.",
            "Total Evaluation Clips: 223 test sequences containing forward falls, backward falls, syncopal collapses, and daily routines.",
            "Sensitivity (Recall): 96.8% (61/63 true fall events successfully detected and alerted).",
            "Specificity: 98.1% (157/160 normal activity clips correctly rejected without false alarms).",
            "Overall Accuracy: 97.5% across all evaluated scenarios."
        ],
        card2_title="Confusion Matrix Breakdown",
        card2_bullets=[
            "True Positives (61): Forward trips (22), Backward slips (19), Lateral collapses (11), Bed roll-offs (9).",
            "False Positives (3): Rapid flop onto soft bed (1), Aggressive floor stretching (1), Sudden slip with recovery (1).",
            "False Negatives (2): Slow slide down a wall with partial furniture occlusion.",
            "True Negatives (157): Walking, Sitting down, Tying shoelaces, Picking up objects, Reclining on couch.",
            "Receiver Operating Characteristic (ROC): Area Under Curve (AUC) = 0.985."
        ],
        takeaway="96.8% sensitivity and 98.1% specificity confirm the system's readiness for real-world geriatric care deployment.",
        card1_color=COLOR_ACCENT_GREEN,
        card2_color=COLOR_ACCENT_AMBER
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 46: Hardware Singularity Edge-Case Solutions
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 13: Verification & Results",
        title="Hardware Singularity Edge-Case Resolutions",
        card_title="Solving Subtle Real-World Production Bugs",
        bullets=[
            "The Sub-Millisecond Windows Clock Singularity: On Windows DirectShow webcam frame 1, frame capture took <1 microsecond, causing now == prev_time -> dt = 0.0 -> 1.0 / 0.0 crash (ZeroDivisionError).",
            "Resolution: Clamped time delta dt = max(1e-4, now - prev_time) and guarded rolling FPS = min(120, 1.0 / max(0.001, avg_dt)).",
            "Webcam Property Anomalies: Windows DirectShow queries often return 0.0 or NaN for CAP_PROP_FPS. Implemented safe validation fallback to 25.0 FPS.",
            "Lazy VideoWriter Resolution Sync: Webcams often report 0 width/height prior to decoding frame 0. VideoWriter now initializes lazily upon reading the first valid frame, matching exact dimensions.",
            "Exception Traceback Logging: Replaced opaque one-line print(exc) with traceback.print_exc() in main.py for clean debugging."
        ],
        takeaway="Eliminating timer singularities and corrupt hardware queries ensures the software never crashes during 24/7 runtime.",
        accent_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 47: Limitations & Future Engineering Horizons
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_dual_card_slide(
        category="Part 14: Challenges & Future Work",
        title="Limitations & Future Engineering Horizons",
        card1_title="Current System Limitations",
        card1_bullets=[
            "Total Physical Occlusion: Cannot track subjects who collapse completely behind solid walls or heavy wardrobes.",
            "Low-Light Conditions: Standard RGB cameras degrade in darkness without infrared (IR) night illumination.",
            "Extreme Distance: Subjects >10 meters from standard webcams yield low-resolution landmark patches.",
            "Complex Bed Exercises: Certain floor physical therapy routines can mimic fall stillness if prolonged."
        ],
        card2_title="Planned Future Extensions",
        card2_bullets=[
            "3D Instant Fall Replay (Rewind Buffer): 5-second in-memory buffer allowing clinicians to scrub and inspect falls in 3D.",
            "Near-Miss & Stumble Prediction: Early detection of balance instability to warn of impending fall risks before impact occurs.",
            "ONNX / OpenVINO Runtime: Further accelerating YOLOv8 on low-power Intel N100 mini-PCs toward 45+ FPS.",
            "WebRTC Browser Dashboard: Streaming dual-panel video directly to mobile browsers over local Wi-Fi networks."
        ],
        takeaway="Future work centers on 3D slow-motion replay buffers, stumble prediction, and WebRTC mobile streaming.",
        card1_color=COLOR_ACCENT_AMBER,
        card2_color=COLOR_ACCENT_CYAN
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Slide 48: Summary of Contributions & Conclusion
    # ─────────────────────────────────────────────────────────────────────────
    builder.add_wide_card_slide(
        category="Part 14: Challenges & Future Work",
        title="Summary of Contributions & Academic Conclusion",
        card_title="Final Research & Engineering Summary",
        bullets=[
            "End-to-End Autonomous Safety: Built a complete, contactless AI fall detection and activity monitoring system operating on commodity hardware.",
            "Edge Performance Achieved: Optimized pipeline throughput from 8.9 FPS to 30.5 FPS sustained on CPU using adaptive pose and resolution tuning.",
            "Anatomical 3D Visualizer: Formulated an extended 35-keypoint topology with central spine, multi-tier joint hierarchy, and depth-sorted rendering.",
            "Robust 12-Factor Decision Engine: Achieved 97.5% overall accuracy with 96.8% sensitivity and 98.1% specificity across clinical benchmarks.",
            "Enterprise Alerting & UX: Integrated two-phase acoustic sirens, non-blocking Telegram/Email push dispatch, and a 5-tier workstation dashboard.",
            "Academic Conclusion: Contactless, edge-deployable AI can provide life-saving geriatric monitoring without compromising user privacy or requiring expensive GPUs."
        ],
        takeaway="Thank You! The system is fully operational, verified across 60 unit tests, and ready for deployment.",
        accent_color=COLOR_ACCENT_GREEN
    )

    builder.save()


if __name__ == "__main__":
    build_all_48_slides()
