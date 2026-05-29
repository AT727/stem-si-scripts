import sys, json, cv2
import numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")

# Check mean brightness of ROI for various frames
for fn in [0, 50, 100, 150, 200, 300, 400, 500, 1000, 2000]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(roi_gray)
    top_mean = np.mean(roi_gray[:100, :])  # Top 100 rows (air region)
    bot_mean = np.mean(roi_gray[-100:, :])  # Bottom 100 rows (water region)
    print(f"Frame {fn}: overall_mean={mean_bright:.1f}, top_100_mean={top_mean:.1f}, bot_100_mean={bot_mean:.1f}")
