import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 60
BASELINE_FRAMES = 100

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
frames_sub = []
for fn in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    frames_sub.append(rg[:, :TAPE_EDGE_COLS])
baseline_profile = np.median(frames_sub, axis=0)
air_ref = float(np.median(baseline_profile[:30]))

search_start = int(0.25 * h)
search_end = int(0.95 * h)

fn = 300
cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
ret, frame = cap.read()
rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
sub = rg[:, :TAPE_EDGE_COLS].astype(np.float32)
air_frame = np.median(sub[:30])
scale = air_ref / air_frame
sub_norm = sub * scale

diff = np.median(np.abs(sub_norm - baseline_profile), axis=1)
k = np.ones(15)/15
dsm = np.convolve(diff, k, mode='same')

print(f"Frame {fn}: air_frame={air_frame:.1f}, scale={scale:.3f}")
print(f"\nRows {search_start}-{search_end} diff_sm:")
for r in range(search_start, search_end):
    if dsm[r] > 8:
        print(f"  row {r}: diff_sm={dsm[r]:.2f}")

grads = np.abs(np.diff(dsm))
reg = dsm[search_start:search_end]
print(f"\nMax reg: {np.max(reg):.2f} at row {search_start+np.argmax(reg)}")
print(f"Max diff (full ROI): {np.max(diff):.2f} at row {np.argmax(diff)}")

# Also check if there's some feature at row 346
print(f"\nsub_norm at rows 340-350:")
for r in [340,342,344,346,348,350]:
    print(f"  row {r}: mean={sub_norm[r].mean():.1f}, median={np.median(sub_norm[r]):.1f}")

print(f"\nbaseline_profile at rows 340-350:")
for r in [340,342,344,346,348,350]:
    print(f"  row {r}: baseline={baseline_profile[r].mean():.1f}")
