import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 57; TAPE_COL_OFFSET = 48
BASELINE_FRAMES = 100
AIR_REF_ROWS = 30; DIFF_SMOOTH = 15
col_slice = slice(TAPE_COL_OFFSET, TAPE_COL_OFFSET+TAPE_EDGE_COLS)
search_start = int(0.25 * h); search_end = int(0.95 * h)

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
frames_sub = []
for fn in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    frames_sub.append(rg[:, col_slice])
bp = np.median(frames_sub, axis=0).astype(np.float32)
air_ref = float(np.median(bp[:AIR_REF_ROWS]))

fn = 400
cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
ret, frame = cap.read()
rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
sub = rg[:, col_slice].astype(np.float32)
sub *= air_ref / np.median(sub[:AIR_REF_ROWS])

diff = np.median(np.abs(sub - bp), axis=1)
k = np.ones(DIFF_SMOOTH)/DIFF_SMOOTH
dsm = np.convolve(diff, k, mode='same')

# Print full diff profile for rows 200-500
print("Frame 400: full diff profile (rows 200-500):")
for r in range(200, 501):
    if diff[r] > 5 or (r >= 425 and r <= 445):
        print(f"  row {r}: diff={diff[r]:.1f}, dsm={dsm[r]:.1f}, sub_mean={sub[r].mean():.1f}, bl_mean={bp[r].mean():.1f}")

print(f"\npeak diff: {np.max(diff):.1f} at row {np.argmax(diff)}")
print(f"peak dsm: {np.max(dsm):.1f} at row {np.argmax(dsm)}")
print(f"search_end dsm: {dsm[search_end]:.1f}")

# Also show water surface region
print(f"\nWater surface should be at row ~430 (6.34cm height):")
for r in range(415, 450):
    print(f"  row {r}: diff={diff[r]:.1f}, dsm={dsm[r]:.1f}, sub={sub[r].mean():.1f}, bl={bp[r].mean():.1f}")
