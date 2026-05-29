import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 15; TAPE_COL_OFFSET = 0
col_slice = slice(TAPE_COL_OFFSET, TAPE_COL_OFFSET+TAPE_EDGE_COLS)
search_start = int(0.25 * h); search_end = int(0.95 * h)

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")

# Test frame-differencing on specific frames
test_frames = [100, 200, 300, 350, 400, 405, 450, 500]

for fn in test_frames:
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
    
    max_reg = np.max(reg)
    argmax_row = search_start + np.argmax(reg)
    
    print(f"Frame {fn}: max(reg)={max_reg:.2f}, argmax_row={argmax_row}, height={(search_end-argmax_row)/77.33:.3f}")
