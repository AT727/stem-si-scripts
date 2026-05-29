import sys
import cv2
import json
import numpy as np

TAPE_EDGE_COLS = 60  # matches analyze.py — leftmost columns used for detection

def _show_scaled(window_name, image, max_width=1280, max_height=720):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    display_w, display_h = int(w * scale), int(h * scale)
    cv2.resizeWindow(window_name, display_w, display_h)
    cv2.imshow(window_name, image)


cap = cv2.VideoCapture(sys.argv[1])
ret, frame = cap.read()
if not ret:
    print(f"Failed to read first frame from: {sys.argv[1]}")
    print(f"Video opened: {cap.isOpened()}")
    cap.release()
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Detected FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Frame resolution: {frame.shape[1]}x{frame.shape[0]}")

if fps < 1:
    fps = float(input("FPS is 0 or invalid. Enter FPS manually: "))

_show_scaled("First Frame - Select ROI around tape column", frame)
print("Draw a rectangle tightly around the measuring tape column.")
print("WARNING: Exclude the right side of the frame (blue foam board).")
roi = cv2.selectROI("First Frame - Select ROI around tape column", frame)
cv2.destroyWindow("First Frame - Select ROI around tape column")

x, y, w, h = map(int, roi)

if w == 0 or h == 0:
    print("ROI selection cancelled. Exiting.")
    sys.exit(1)

if w > 200:
    print(f"WARNING: ROI width is {w}px — may include non-tape regions.")
if w > 60:
    print(f"NOTE: ROI width {w}px is wider than the tape edge ({TAPE_EDGE_COLS}px used by analyze.py).")

roi_frame = frame[y:y+h, x:x+w]
_show_scaled("ROI - Click two points on the tape", roi_frame)

clicks = []

def mouse_callback(event, x_roi, y_roi, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x_roi, y_roi))
        print(f"Click {len(clicks)}: ({x_roi}, {y_roi})")
        display = roi_frame.copy()
        for i, (cx, cy) in enumerate(clicks):
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.circle(display, (cx, cy), 5, color, -1)
            cv2.putText(display, str(i + 1), (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("ROI - Click two points on the tape", display)

cv2.setMouseCallback("ROI - Click two points on the tape", mouse_callback)

print("\nClick two points on the measuring tape:")
print("  Click 1: the zero/still-water baseline mark")
print("  Click 2: a known reference point N cm above it")
print("(Press any key after both clicks are placed)\n")

cv2.waitKey(0)
cv2.destroyWindow("ROI - Click two points on the tape")

if len(clicks) < 2:
    print("Need exactly 2 clicks. Restart calibration.")
    sys.exit(1)

point1 = clicks[0]
point2 = clicks[1]

ref_cm = float(input("Enter the real-world distance in cm between the two clicked points: "))

pixel_distance = abs(point1[1] - point2[1])

if pixel_distance < 5:
    print(f"ERROR: Points are only {pixel_distance}px apart — too close for accurate ratio.")
    print("Re-run calibration and select points further apart.")
    sys.exit(1)

pixels_per_cm = pixel_distance / ref_cm
baseline_y = point1[1]

calibration = {
    "pixels_per_cm": round(pixels_per_cm, 2),
    "roi": {"x": x, "y": y, "w": w, "h": h},
    "baseline_y": baseline_y,
    "fps": fps
}

with open("calibration.json", "w") as f:
    json.dump(calibration, f, indent=2)

print(f"\nComputed pixels_per_cm: {pixels_per_cm:.2f}")

with open("calibration.json", "r") as f:
    saved = json.load(f)
print("Saved calibration.json — valid JSON confirmed")

confirm = roi_frame.copy()
for i, (cx, cy) in enumerate(clicks):
    color = (0, 255, 0) if i == 0 else (255, 0, 0)
    cv2.circle(confirm, (cx, cy), 5, color, -1)
    cv2.putText(confirm, str(i + 1), (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.line(confirm, (0, baseline_y), (w - 1, baseline_y), (0, 255, 255), 1)
cv2.putText(confirm, "baseline", (5, baseline_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

_show_scaled("Calibration Verification", confirm)
print("Close the window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
