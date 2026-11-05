"""
Unit and integration tests for multi-person detection, tracking,
non-destructive track suppression, and interactive subject selection.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np

from person_tracker import PersonTracker
from yolo_detector import TrackedBox
from visualizer_3d import SkeletonVisualizer3D
from dashboard import Dashboard, draw_person_box


class TestMultiPersonTracking(unittest.TestCase):

    def setUp(self):
        self.tracker = PersonTracker(max_missing=15)
        self.vis_3d = SkeletonVisualizer3D(width=320, height=480)
        self.dash = Dashboard()

    def test_proximity_not_suppressed(self):
        """
        Two persons standing near each other with IoMIN ~ 0.38 and IoU ~ 0.22
        must NOT be suppressed (resolves the critical real-life multi-person bug).
        """
        # Box 1: [100, 50, 200, 350] (area = 30000)
        # Box 2: [160, 60, 260, 360] (area = 30000)
        # Intersection: [160, 60, 200, 350] -> 40 x 290 = 11600
        # IoMIN = 11600 / 30000 = 0.387 (previously killed by iomin >= 0.25)
        b1 = TrackedBox(track_id=1, x1=100, y1=50, x2=200, y2=350, confidence=0.85)
        b2 = TrackedBox(track_id=2, x1=160, y1=60, x2=260, y2=360, confidence=0.82)

        tracks = self.tracker.update([b1, b2])

        # Both persons must be retained
        self.assertEqual(len(tracks), 2)
        tracked_ids = [t[0] for t in tracks]
        self.assertIn(1, tracked_ids)
        self.assertIn(2, tracked_ids)

    def test_three_simultaneous_persons(self):
        """Verify that three people in different areas of the frame are all tracked."""
        b1 = TrackedBox(track_id=1, x1=50, y1=100, x2=150, y2=400, confidence=0.90)
        b2 = TrackedBox(track_id=2, x1=250, y1=100, x2=350, y2=400, confidence=0.88)
        b3 = TrackedBox(track_id=3, x1=450, y1=100, x2=550, y2=400, confidence=0.85)

        tracks = self.tracker.update([b1, b2, b3])
        self.assertEqual(len(tracks), 3)
        self.assertEqual({t[0] for t in tracks}, {1, 2, 3})

    def test_true_duplicate_suppression(self):
        """
        Verify that a true duplicate box of the SAME person (IoU >= 0.70)
        is properly suppressed.
        """
        b1 = TrackedBox(track_id=1, x1=100, y1=50, x2=200, y2=350, confidence=0.90)
        b2_dup = TrackedBox(track_id=2, x1=105, y1=52, x2=205, y2=352, confidence=0.75)

        tracks = self.tracker.update([b1, b2_dup])
        # Duplicate should be suppressed
        self.assertEqual(len(tracks), 1)

    def test_nested_torso_subbox_suppression(self):
        """
        Verify that an accidental torso-only sub-box inside a full body box
        is suppressed as a duplicate.
        """
        b1_full = TrackedBox(track_id=1, x1=100, y1=50, x2=200, y2=350, confidence=0.90)
        b2_torso = TrackedBox(track_id=2, x1=105, y1=60, x2=195, y2=200, confidence=0.70)

        tracks = self.tracker.update([b1_full, b2_torso])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0][0], 1)

    def test_missing_track_aging_and_reactivation(self):
        """
        Tracks missing from the current frame must not appear as active ghost boxes,
        but must preserve state for ID inheritance upon re-detection.
        """
        b1 = TrackedBox(track_id=1, x1=100, y1=50, x2=200, y2=350, confidence=0.85)
        b2 = TrackedBox(track_id=2, x1=300, y1=50, x2=400, y2=350, confidence=0.85)

        # Frame 1: both seen
        t1 = self.tracker.update([b1, b2])
        self.assertEqual(len(t1), 2)

        # Add custom state property
        self.tracker._states[2]["custom_flag"] = "preserved"

        # Frame 2: Person 2 missing (temporarily occluded)
        t2 = self.tracker.update([b1])
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0][0], 1)
        # Person 2 state is retained internally with missing=1
        self.assertIn(2, self.tracker._states)
        self.assertEqual(self.tracker._states[2]["missing"], 1)

        # Frame 3: Person 2 reappears
        t3 = self.tracker.update([b1, b2])
        self.assertEqual(len(t3), 2)
        self.assertEqual(self.tracker._states[2]["custom_flag"], "preserved")
        self.assertEqual(self.tracker._states[2]["missing"], 0)

    def test_3d_visualizer_person_id_hud(self):
        """Verify that 3D visualizer renders without error with person_id parameter."""
        canvas = self.vis_3d.render(
            world_landmarks=None,
            activity_label="Walking",
            confidence=0.92,
            is_fall=False,
            person_id=3,
        )
        self.assertEqual(canvas.shape, (480, 320, 3))
        self.assertTrue(np.any(canvas > 0))

    def test_dashboard_multi_person_rendering(self):
        """Verify dashboard drawing with multiple active persons and distinct statuses."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state1 = {
            "activity_label": "Standing",
            "activity_confidence": 0.95,
            "fall_detected": False,
            "fall_score": 0.05,
            "severity": "NONE",
            "person_id": 1,
        }
        state2 = {
            "activity_label": "Walking",
            "activity_confidence": 0.91,
            "fall_detected": False,
            "fall_score": 0.12,
            "severity": "NONE",
            "person_id": 2,
        }
        tracked = [
            (1, [80, 100, 180, 380], state1),
            (2, [220, 100, 320, 380], state2),
        ]

        draw_person_box(frame, tracked[0][1], pid=1, state=state1)
        draw_person_box(frame, tracked[1][1], pid=2, state=state2)
        out = self.dash.draw(frame, active_persons=2, fps=30.0, summary_state=state1, tracked_persons=tracked)

        self.assertEqual(out.shape, (480, 640, 3))
        self.assertTrue(np.any(out > 0))


if __name__ == "__main__":
    unittest.main()
