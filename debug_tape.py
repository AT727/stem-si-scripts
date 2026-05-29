import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 57; TAPE_COL_OFFSET = 48
BASELINE_FRAMES = 100; BASELINE_FRAMES = 100
MIN_DIFF_CONFIDENCE = 10; DIFF_SMOOTH = 15
AIR_REF_ROWS = 30
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

for fn in [350, 400, 450, 500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    sub = rg[:, col_slice].astype(np.float32)
    air_frame = np.median(sub[:AIR_REF_ROWS])
    sub *= air_ref / air_frame
    
    diff = np.median(np.abs(sub - bp), axis=1)
    k = np.ones(DIFF_SMOOTH)/DIFF_SMOOTH
    dsm = np.convolve(diff, k, mode='same')
    
    reg = dsm[search_start:search_end]
    print(f"Frame {fn}: peak diff={np.max(dsm):.1f} at row {np.argmax(dsm)}")
    print(f"  reg range: [{reg.min():.1f}, {reg.max():.1f}]")
    grads = np.abs(np.diff(reg))
    maxgrad_row = search_start + np.argmax(grads)
    print(f"  max grad row: {maxgrad_row}, height={(search_end-maxgrad_row)/77.33:.3f}")
    
    # Show diff around detection
    for r in range(maxgrad_row-3, maxgrad_row+4):
        print(f"  row {r}: diff={diff[r]:.1f}, dsm={dsm[r]:.1f}")
    print()
