import sys, json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)

pixels_per_cm = cal["pixels_per_cm"]
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

TAPE_EDGE_COLS = 60
BASELINE_FRAMES = 100
DIFF_SMOOTH = 15
MIN_DIFF_CONFIDENCE = 4
SEARCH_START_FRAC = 0.25
SEARCH_END_FRAC = 0.95

edge_cols = min(TAPE_EDGE_COLS, w)
search_start = int(h * SEARCH_START_FRAC)
search_end = int(h * SEARCH_END_FRAC)

print(f"h={h}, search_start={search_start}, search_end={search_end}, baseline_y={search_end}")
print(f"pixels_per_cm={pixels_per_cm}")
print(f"Max possible height: {(search_end - search_start) / pixels_per_cm:.3f} cm")

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
baseline_frames = []
for i in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    baseline_frames.append(roi_gray)

baseline_stack = np.stack(baseline_frames, axis=0)[:, :, :edge_cols]
baseline_profile = np.median(baseline_stack, axis=0).astype(np.float32)

# Check frame 150 (a wave frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
ret, frame = cap.read()
roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

sub = roi_gray[:, :edge_cols].astype(np.float32)
bl = baseline_profile
diff = np.median(np.abs(sub - bl), axis=1)
k = np.ones(DIFF_SMOOTH) / DIFF_SMOOTH
diff_sm = np.convolve(diff, k, mode='same')
reg = diff_sm[search_start:search_end]

print(f"\nFrame 150 debug:")
print(f"  diff shape: {diff.shape}")
print(f"  reg shape: {reg.shape}")
print(f"  max(reg): {np.max(reg):.2f}")
print(f"  min(reg): {np.min(reg):.2f}")
print(f"  max(reg) < MIN_DIFF_CONFIDENCE? {np.max(reg) < MIN_DIFF_CONFIDENCE}")

gradients = np.abs(np.diff(reg))
print(f"  gradients shape: {gradients.shape}")
print(f"  max(gradients): {np.max(gradients):.2f}")
print(f"  argmax(gradients): {np.argmax(gradients)}")

surface_y = search_start + np.argmax(gradients) if len(gradients) > 0 else search_end
if np.max(reg) < MIN_DIFF_CONFIDENCE:
    surface_y = search_end
print(f"  surface_y: {surface_y}")
print(f"  displacement: {search_end - surface_y}")
print(f"  height: {(search_end - surface_y) / pixels_per_cm:.3f} cm")

# Also print first and last 10 values of reg
print(f"\n  reg[0:10]: {reg[:10].round(1)}")
print(f"  reg[-10:]: {reg[-10:].round(1)}")
print(f"\n  diff_sm[0:20]: {diff_sm[:20].round(1)}")
print(f"  diff_sm[search_start-5:search_start+15]: {diff_sm[search_start-5:search_start+15].round(1)}")
