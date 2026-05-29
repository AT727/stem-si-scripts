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
MIN_DIFF_CONFIDENCE = 10

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

def detect(roi_gray):
    sub = roi_gray[:, :edge_cols].astype(np.float32)
    bl = baseline_profile
    diff = np.median(np.abs(sub - bl), axis=1)
    k = np.ones(DIFF_SMOOTH) / DIFF_SMOOTH
    diff_sm = np.convolve(diff, k, mode='same')
    reg = diff_sm[search_start:search_end]
    
    if np.max(reg) < MIN_DIFF_CONFIDENCE:
        return search_end, reg, None
    
    gradients = np.abs(np.diff(reg))
    return search_start + np.argmax(gradients), reg, gradients

for fn in [193, 240, 366, 400, 600]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    surface_y, reg, grad = detect(roi_gray)
    height = (search_end - surface_y) / 77.33
    print(f"Frame {fn}: surface_y={surface_y}, height={height:.3f} cm, max(reg)={np.max(reg):.1f}")
    if grad is not None:
        max_grad_idx = np.argmax(grad)
        print(f"  argmax(grad) idx={max_grad_idx}, row={search_start + max_grad_idx}")
        # Show region around argmax
        start = max(0, max_grad_idx - 5)
        end = min(len(grad), max_grad_idx + 5)
        print(f"  grad[{start}:{end}]: {[f'{g:.2f}' for g in grad[start:end]]}")
        print(f"  reg[{start}:{end+1}]: {[f'{r:.2f}' for r in reg[start:end+1]]}")
