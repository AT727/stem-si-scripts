import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 60; BASELINE_FRAMES = 100
MIN_DIFF_CONFIDENCE = 10; DIFF_SMOOTH = 15

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
frames_sub = []
for fn in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    frames_sub.append(rg[:, :TAPE_EDGE_COLS])
bp = np.median(frames_sub, axis=0)
air_ref = float(np.median(bp[:30]))
search_start = int(0.25 * h); search_end = int(0.95 * h)

fn = 300
cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
ret, frame = cap.read()
rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
sub = rg[:, :TAPE_EDGE_COLS].astype(np.float32)
air_frame = np.median(sub[:30])
sub *= air_ref / air_frame

diff = np.median(np.abs(sub - bp), axis=1)
k = np.ones(DIFF_SMOOTH)/DIFF_SMOOTH
dsm = np.convolve(diff, k, mode='same')
reg = dsm[search_start:search_end]
print(f"reg range: [{reg.min():.2f}, {reg.max():.2f}]")

grads = np.abs(np.diff(reg))
# Find top 5 gradient positions
top5 = np.argsort(grads)[-5:][::-1]
print(f"Top 5 gradient positions (reg-relative): {top5}")
for i, g in enumerate(top5):
    row = search_start + g
    print(f"  #{i}: row {row}, grad={grads[g]:.4f}, reg_at_row={dsm[row]:.2f}, reg_before={dsm[row-1]:.2f}")
    print(f"       diff values around row: {diff[row-2:row+3].round(1)}")
    hgt = (search_end - row) / 77.33
    print(f"       height = {hgt:.3f} cm")

# Also check for a sharp rise near row 346
print(f"\nDiff_sm profile around detection (row 346):")
for r in range(340, 355):
    print(f"  row {r}: diff={diff[r]:.2f}, dsm={dsm[r]:.2f}, bl={bp[r].mean():.1f}, frame={sub[r].mean():.1f}")
