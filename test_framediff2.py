import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 15; TAPE_COL_OFFSET = 0
col_slice = slice(TAPE_COL_OFFSET, TAPE_COL_OFFSET+TAPE_EDGE_COLS)
search_start = int(0.25 * h); search_end = int(0.95 * h)

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")

for fn in [400, 405, 450, 500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, curr = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn - 1)
    ret, prev = cap.read()
    
    curr_gray = cv2.cvtColor(curr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    
    diff = np.median(np.abs(curr_gray[:, col_slice].astype(np.float32) - 
                            prev_gray[:, col_slice].astype(np.float32)), axis=1)
    
    k = np.ones(51)/51
    dsm = np.convolve(diff, k, mode='same')
    reg = dsm[search_start:search_end]
    
    peak = np.max(reg)
    print(f"Frame {fn}: peak={peak:.2f}")
    # Full profile in search window
    for r in range(search_start, search_end):
        if dsm[r] > peak * 0.3:  # show where diff is significant
            if r < search_start + 300:  # only first 300 rows of search window
                print(f"  row {r}: dsm={dsm[r]:.2f}")
    print(f"  argmax: row={search_start+np.argmax(reg)}, height={(search_end-search_start-np.argmax(reg))/77.33:.3f}")
    
    # Threshold crossing approach
    threshold = peak * 0.5
    crossings = np.where(reg >= threshold)[0]
    if len(crossings) > 0:
        cross_row = search_start + crossings[0]
        print(f"  50% crossing: row={cross_row}, height={(search_end-cross_row)/77.33:.3f}")
    else:
        print(f"  50% crossing: none")
    print()
