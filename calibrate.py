import sys
import ctypes
import cv2
import json
import numpy as np

def calibrate(video_path, dist_cm=None):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        return

    # Scale display to fit screen for ROI selection
    img_h, img_w = frame.shape[:2]
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    max_display_w = int(screen_w * 0.95)
    max_display_h = int(screen_h * 0.85)
    scale = min(max_display_w / img_w, max_display_h / img_h, 1.0)

    if scale < 1.0:
        display_w = int(img_w * scale)
        display_h = int(img_h * scale)
        display_frame = cv2.resize(frame, (display_w, display_h))
    else:
        display_frame = frame

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    roi = cv2.selectROI("Calibration", display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Calibration")
    x, y, w, h = roi
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)
    
    # User clicks
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

    print("\nClick 3 points in the zoomed window (in order):")
    print("  Click 1: Still-water baseline (this is your zero reference)")
    print("  Click 2: A known reference point some distance above Click 1")
    print("  Click 3: The wave ceiling (the highest point you expect waves to reach)")
    print()

    click_frame = frame[y:y+h, x:x+w].copy()
    instruction = "Click 3 points in order (see terminal for descriptions). Press any key when done."
    cv2.putText(click_frame, instruction, (10, click_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.namedWindow("Click Points")
    cv2.setMouseCallback("Click Points", mouse_callback)
    cv2.imshow("Click Points", click_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) < 3:
        print("Need 3 clicks: baseline, ref, ceiling")
        return

    base_y = points[0][1] + y
    ref_y = points[1][1] + y
    ceiling_y = points[2][1] + y

    if dist_cm is None:
        dist_cm = float(input("Enter distance between points 1 and 2 (cm): "))
    pixels_per_cm = abs(base_y - ref_y) / dist_cm
    
    headroom_cm = (base_y - ceiling_y) / pixels_per_cm
    print(f"Wave ceiling headroom: {headroom_cm:.1f} cm above baseline")
    
    cal = {
        "pixels_per_cm": pixels_per_cm,
        "baseline_y": base_y,
        "wave_ceiling_y": ceiling_y,
        "roi": {"x": x, "y": y, "w": w, "h": h},
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "edge_cols": 60
    }
    
    with open("calibration.json", "w") as f:
        json.dump(cal, f, indent=2)
    print("Saved to calibration.json")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <video_path> [dist_cm]")
        sys.exit(1)
    video_path = sys.argv[1]
    dist_cm = float(sys.argv[2]) if len(sys.argv) > 2 else None
    calibrate(video_path, dist_cm)
