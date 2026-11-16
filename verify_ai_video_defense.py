import os
import sys
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "visualization"))

from fall_detector import FallDetector

def run_fall_detection_test(video_path):
    video_name = os.path.basename(video_path)
    print("\n" + "="*70)
    print(f"TESTING END-TO-END PIPELINE ON: {video_name}")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    detector = FallDetector(
        display=False,
        enable_3d=False,
        enable_alerts=False,
        enable_ai_video_defense=True,
        video_source_path=video_path,
    )
    
    frame_idx = 0
    confirmed_falls = 0
    suppressed_falls = 0
    authenticity_verdicts = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        now = frame_idx / fps
        
        out_frame, telemetry = detector.process_single_frame(frame, now=now)
        auth = telemetry.get("video_authenticity", {})
        verdict = auth.get("verdict", "UNKNOWN")
        authenticity_verdicts.append(verdict)
        
        for p in telemetry.get("persons", []):
            if p.get("fall_detected"):
                confirmed_falls += 1
                
    cap.release()
    
    final_auth = detector.ai_detector.get_aggregate_verdict() if detector.ai_detector else None
    
    print("\n" + "-"*50)
    print(f"RESULTS FOR {video_name}:")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Final Authenticity Verdict: {final_auth.final_result if final_auth else 'N/A'}")
    print(f"Authenticity Confidence: {final_auth.confidence*100:.1f}%")
    print(f"AI Probability: {final_auth.ai_probability:.3f} | Real Probability: {final_auth.real_probability:.3f}")
    print(f"Dashboard Total Confirmed Falls: {detector.dashboard.total_falls}")
    print(f"Key Detection Reasons: {final_auth.reasons[:3] if final_auth else []}")
    print("-"*50)
    
    return {
        "video": video_name,
        "verdict": final_auth.final_result if final_auth else "UNKNOWN",
        "confidence": final_auth.confidence if final_auth else 0.0,
        "total_falls": detector.dashboard.total_falls,
    }

if __name__ == "__main__":
    real_path = os.path.join("Demo Videos", "real human fallling.mp4")
    ai_path = os.path.join("Demo Videos", "ai_made.mp4")
    
    real_res = run_fall_detection_test(real_path)
    ai_res = run_fall_detection_test(ai_path)
    
    print("\n" + "#"*70)
    print("FINAL HEAD-TO-HEAD COMPARISON:")
    print(f"1. Real Human Falling ({real_res['video']}):")
    print(f"   -> Authenticity: {real_res['verdict']} (Confidence: {real_res['confidence']*100:.1f}%)")
    print(f"   -> Fall Detected & Confirmed: {real_res['total_falls']} fall(s)")
    print()
    print(f"2. AI Made Video ({ai_res['video']}):")
    print(f"   -> Authenticity: {ai_res['verdict']} (Confidence: {ai_res['confidence']*100:.1f}%)")
    print(f"   -> Fall Detected & Confirmed: {ai_res['total_falls']} fall(s) (Suppressed)")
    print("#"*70)
    
    # Assertions required by user:
    # 1 must show fall (real human falling)
    # 1 must show AI made (ai_made.mp4) and suppress false fall alarm
    assert real_res["verdict"] == "REAL", f"Real video must be classified as REAL, got {real_res['verdict']}"
    assert real_res["total_falls"] >= 1, f"Real video must detect fall, got {real_res['total_falls']}"
    
    assert ai_res["verdict"] == "AI/FAKE", f"AI video must be classified as AI/FAKE, got {ai_res['verdict']}"
    assert ai_res["total_falls"] == 0, f"AI video must NOT confirm fall (should be 0 falls), got {ai_res['total_falls']}"
    
    print("\n SUCCESS: 100% OF OBJECTIVES MET! REAL VIDEO CONFIRMED FALL, AI VIDEO SUPPRESSED!")
