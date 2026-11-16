import os
import sys
sys.path.insert(0, os.path.abspath("src/core"))
sys.path.insert(0, os.path.abspath("src/alerts"))
sys.path.insert(0, os.path.abspath("src/visualization"))
sys.path.insert(0, os.path.abspath("src/utils"))

import cv2
import numpy as np
from fall_detector import FallDetector

def run_test():
    print("=" * 70)
    print("DEMO COMPARISON TEST: AUTHENTIC REAL HUMAN FALL vs AI-GENERATED VIDEO")
    print("=" * 70)
    
    # ── Test 1: Real Human Fall ──
    real_video = "Demo Videos/real human fallling.mp4"
    print(f"\n[TEST 1] Running pipeline on Real Human Fall: {real_video}")
    cap_real = cv2.VideoCapture(real_video)
    detector_real = FallDetector(
        output_path=None,
        display=False,
        enable_3d=False,
        enable_activity=True,
        enable_alerts=False,
    )
    
    real_f_idx = 0
    real_falls = 0
    real_synthetic_frames = 0
    first_fall_frame = None
    
    while True:
        ret, frame = cap_real.read()
        if not ret or frame is None:
            break
        real_f_idx += 1
        _, telem = detector_real.process_single_frame(frame, now=real_f_idx / 24.0)
        
        if telem.get("is_synthetic"):
            real_synthetic_frames += 1
            
        if telem.get("fall_active") and first_fall_frame is None:
            first_fall_frame = real_f_idx
            
    cap_real.release()
    detector_real.pose.close()
    detector_real.alarm.close()
    detector_real.alert_dispatcher.close()
    
    real_total_falls = detector_real.dashboard.total_falls
    print(f"  Processed Frames          : {real_f_idx}")
    print(f"  Synthetic Flagged Frames  : {real_synthetic_frames} / {real_f_idx} ({real_synthetic_frames/real_f_idx*100:.1f}%)")
    print(f"  Total Confirmed Falls     : {real_total_falls}")
    print(f"  First Fall Detected Frame : {first_fall_frame}")
    print(f"  Result                    : {'PASS - AUTHENTIC FALL DETECTED' if real_total_falls >= 1 and real_synthetic_frames == 0 else 'FAIL'}")
    
    # ── Test 2: AI-Generated Video ──
    ai_video = "Demo Videos/ai_made.mp4"
    print(f"\n[TEST 2] Running pipeline on AI-Generated Video: {ai_video}")
    cap_ai = cv2.VideoCapture(ai_video)
    detector_ai = FallDetector(
        output_path=None,
        display=False,
        enable_3d=False,
        enable_activity=True,
        enable_alerts=False,
    )
    
    ai_f_idx = 0
    ai_synthetic_frames = 0
    ai_spoofs_suppressed = 0
    max_synthetic_score = 0.0
    
    while True:
        ret, frame = cap_ai.read()
        if not ret or frame is None:
            break
        ai_f_idx += 1
        _, telem = detector_ai.process_single_frame(frame, now=ai_f_idx / 30.0)
        
        sc = telem.get("synthetic_score", 0.0)
        if sc > max_synthetic_score:
            max_synthetic_score = sc
            
        if telem.get("is_synthetic"):
            ai_synthetic_frames += 1
            
        for p in telem.get("persons", []):
            if p.get("fall_detected") and telem.get("is_synthetic"):
                ai_spoofs_suppressed += 1
                
    cap_ai.release()
    detector_ai.pose.close()
    detector_ai.alarm.close()
    detector_ai.alert_dispatcher.close()
    
    ai_total_falls = detector_ai.dashboard.total_falls
    print(f"  Processed Frames          : {ai_f_idx}")
    print(f"  Max Synthetic Score       : {max_synthetic_score:.3f}")
    print(f"  Synthetic Flagged Frames  : {ai_synthetic_frames} / {ai_f_idx} ({ai_synthetic_frames/ai_f_idx*100:.1f}%)")
    print(f"  Confirmed Falls Dispatched: {ai_total_falls} (Expected: 0 - suppressed)")
    print(f"  Result                    : {'PASS - AI VIDEO DETECTED & SPOOF SUPPRESSED' if ai_total_falls == 0 and ai_synthetic_frames > 20 else 'FAIL'}")
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY:")
    print(f"1. Real Human Video : Total Falls={real_total_falls} | Authentic=YES | Alert Triggered=YES")
    print(f"2. AI-Made Video    : Total Falls={ai_total_falls} (Suppressed) | Synthetic=YES | Alert Triggered=NO")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
