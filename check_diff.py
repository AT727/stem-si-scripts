import sys, json, cv2
import numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

TAPE_EDGE_COLS = 60
BASELINE_FRAMES = 100
DIFF_SMOOTH = 15
SEARCH_START_FRAC = 0.25
SEARCH_END_FRAC = 0.95

edge_cols = min(TAPE_EDGE_COLS, w)
search_start = int(h * SEARCH_START_FRAC)
search_end = int(h * SEARCH_END_FRAC)

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
baseline_frames = []
for i in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    baseline_frames.append(roi_gray)

baseline_stack = np.stack(baseline_frames, axis=0)[:, :, :edge_cols]
baseline_profile = np.median(baseline_stack, axis=0).astype(np.float32)

# Check frames: 150 (first wave), 190 (false detection), 500 (mid wave)
for fn in [150, 190, 500, 1000]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    sub = roi_gray[:, :edge_cols].astype(np.float32)
    diff = np.median(np.abs(sub - baseline_profile), axis=1)
    k = np.ones(DIFF_SMOOTH) / DIFF_SMOOTH
    diff_sm = np.convolve(diff, k, mode='same')
    reg = diff_sm[search_start:search_end]
    
    # Where are the top 5 gradient peaks?
    gradients = np.abs(np.diff(reg))
    top5 = np.argsort(gradients)[-5:][::-1]
    
    print(f"Frame {fn}: max(reg)={np.max(reg):.1f}, mean(reg)={np.mean(reg):.1f}")
    print(f"  Top 5 gradient positions (ROI-relative): {[search_start + i for i in top5]}")
    print(f"  Diff at those positions: {[reg[i] for i in top5]}")
    print(f"  Heights if detected there: {[(search_end - (search_start + i)) / 77.33 for i in top5]}")
