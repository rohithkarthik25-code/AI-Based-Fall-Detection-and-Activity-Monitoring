"""
Quick test script to verify your Telegram and Email alert dispatch.
Sends a test emergency alert with an annotated test snapshot.
"""

import time
import numpy as np
import cv2
from alert_dispatcher import AlertDispatcher


def main():
    print("=" * 60)
    print("Testing Instant Alert Dispatch (Telegram & Email)...")
    print("=" * 60)

    dispatcher = AlertDispatcher(config_path="alert_config.json")

    print(f"Simulation Mode : {dispatcher.config.get('simulation_mode')}")
    print(f"Telegram Active : {dispatcher.config.get('telegram', {}).get('enabled')}")
    print(f"Email Active    : {dispatcher.config.get('email', {}).get('enabled')}")
    print("-" * 60)

    # Create a test frame
    test_frame = np.full((480, 640, 3), 35, dtype=np.uint8)
    # Draw simple visual test pattern
    cv2.putText(test_frame, "AI FALL DETECTION SYSTEM", (80, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(test_frame, "LIVE TEST ALERT DISPATCH", (110, 260), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(test_frame, "Subject: Person #1 | Severity: SEVERE | Score: 0.95", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    print("[1/2] Triggering Test Fall Alert...")
    dispatched = dispatcher.dispatch_fall_alert(
        frame=test_frame,
        person_id=1,
        severity="SEVERE (TEST)",
        score=0.95,
        duration=2.5,
        activity="Fallen",
        location="Live Testing Environment",
    )

    if dispatched:
        print("Alert task submitted to background worker.")
    else:
        print("Alert was throttled or disabled.")

    # Allow worker thread time to connect and complete network transfer
    print("[2/2] Awaiting network response from Telegram and SMTP server...")
    time.sleep(5)

    dispatcher.close()
    print("=" * 60)
    print("Test complete. Check your Telegram chat and Email inbox!")
    print("=" * 60)


if __name__ == "__main__":
    main()

