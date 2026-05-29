import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
BASELINE_FRAMES = 100; AIR_REF_ROWS = 30
search_start = int(0.25 * h); search_end = int(0.95 * h)

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")

# Build baseline on columns 0-10 (background, max signal)
col_slice1 = slice(0, 10)
col_slice2 = slice(48, 105)  # tape

for col_desc, col_slice in [("cols 0-9 (bg)", col_slice1), ("cols 48-104 (tape)", col_slice2)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    bl_frames = []
    for fn in range(BASELINE_FRAMES):
        ret, frame = cap.read()
        rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        bl_frames.append(rg[:, col_slice])
    bp = np.median(bl_frames, axis=0)
    air_ref = np.median(bp[:AIR_REF_ROWS])
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 400)
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    sub = rg[:, col_slice].astype(np.float32)
    air_frame = np.median(sub[:AIR_REF_ROWS])
    sub *= air_ref / air_frame
    
    # Direct brightness (no baseline subtraction)
    profile = np.median(sub, axis=1)
    k = np.ones(15)/15
    sm = np.convolve(profile, k, mode='same')
    
    # Also diff approach
    diff = np.median(np.abs(sub - bp), axis=1)
    dsm = np.convolve(diff, k, mode='same')
    
    print(f"\n{col_desc} at Frame 400:")
    # Check around water surface (row 430)
    for r in [420, 425, 430, 435, 440]:
        print(f"  row {r}: profile={profile[r]:.1f}, sm={sm[r]:.1f}, diff={diff[r]:.1f}, dsm={dsm[r]:.1f}")
    
    # Gradient of smoothed profile
    grad = np.diff(sm)
    grad_window = grad[search_start:search_end-1]
    max_grad_row = search_start + np.argmax(-grad_window)  # max downward gradient
    print(f"  Peak brightness gradient at row {max_grad_row}: grad={-grad_window[np.argmax(-grad_window)]:.2f}")
