import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 10; TAPE_COL_OFFSET = 0
col_slice = slice(TAPE_COL_OFFSET, TAPE_COL_OFFSET+TAPE_EDGE_COLS)
BASELINE_FRAMES = 100; AIR_REF_ROWS = 30
search_start = int(0.25 * h); search_end = int(0.95 * h)

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")

# Build baseline
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
bl_frames = []
for fn in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    bl_frames.append(rg[:, col_slice])
bp = np.median(bl_frames, axis=0).astype(np.float32)
air_ref = np.median(bp[:AIR_REF_ROWS])

# Get prev frame
cap.set(cv2.CAP_PROP_POS_FRAMES, BASELINE_FRAMES - 1)
ret, frame = cap.read()
prev = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

# Test hybrid approach
test_frames = [200, 300, 400, 430, 500, 800, 1000, 1500]
for fn in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    
    sub = rg[:, col_slice].astype(np.float32)
    air_frame = np.median(sub[:AIR_REF_ROWS])
    if air_ref > 0 and air_frame > 0:
        sub *= air_ref / air_frame
    
    prev_sub = prev[:, col_slice].astype(np.float32)
    prev_air = np.median(prev_sub[:AIR_REF_ROWS])
    if air_ref > 0 and prev_air > 0:
        prev_sub *= air_ref / prev_air
    
    # Baseline diff
    base_diff = np.median(np.abs(sub - bp), axis=1)
    
    # Frame diff (motion)
    motion = np.median(np.abs(sub - prev_sub), axis=1)
    
    # Combined
    combined = base_diff * motion
    # Or min
    min_comb = np.minimum(base_diff, motion)
    
    k = np.ones(51)/51
    base_sm = np.convolve(base_diff, k, mode='same')
    motion_sm = np.convolve(motion, k, mode='same')
    combined_sm = np.convolve(combined, k, mode='same')
    min_sm = np.convolve(min_comb, k, mode='same')
    
    print(f"Frame {fn}:")
    for name, reg, peak in [("base  ", base_sm[search_start:search_end], np.max(base_sm[search_start:search_end])),
                            ("motion", motion_sm[search_start:search_end], np.max(motion_sm[search_start:search_end])),
                            ("comb  ", combined_sm[search_start:search_end], np.max(combined_sm[search_start:search_end])),
                            ("min   ", min_sm[search_start:search_end], np.max(min_sm[search_start:search_end]))]:
        if peak < 4:
            print(f"  {name}: peak={peak:.1f} (not detected)")
        else:
            # Gradient argmax
            grads = np.abs(np.diff(reg))
            argmax = search_start + np.argmax(grads)
            hgt = (search_end - argmax) / 77.33
            # Also try max reg (not gradient)
            argmax2 = search_start + np.argmax(reg)
            hgt2 = (search_end - argmax2) / 77.33
            print(f"  {name}: peak={peak:.1f}, grad_row={argmax}({hgt:.3f}), max_row={argmax2}({hgt2:.3f})")
    print()
