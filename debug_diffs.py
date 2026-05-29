import json, cv2, sys, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 60
BASELINE_FRAMES = 100

# Reconstruct baseline
cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
frames_sub = []
for fn in range(BASELINE_FRAMES):
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    frames_sub.append(rg[:, :TAPE_EDGE_COLS])
baseline_profile = np.median(frames_sub, axis=0)
air_ref = float(np.median(baseline_profile[:30]))

# Debug specific frames
for fn in [145, 146, 147, 148, 149, 150, 155, 160, 170, 180, 190, 200, 300, 350]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    sub = rg[:, :TAPE_EDGE_COLS].astype(np.float32)
    
    # With normalization
    air_frame = np.median(sub[:30])
    scale = air_ref / air_frame if air_frame > 0 else 1
    sub_norm = sub * scale
    
    diff_norm = np.median(np.abs(sub_norm - baseline_profile), axis=1)
    diff_raw = np.median(np.abs(sub - baseline_profile), axis=1)
    
    # Search window
    search_start = int(0.25 * h)
    search_end = int(0.95 * h)
    
    print(f"Frame {fn}:")
    print(f"  air_frame={air_frame:.1f}, scale={scale:.3f}")
    print(f"  diff_norm top 10 rows: {diff_norm[:10].round(1)}")
    print(f"  diff_norm search_start[{search_start}]: {diff_norm[search_start]:.1f}")
    print(f"  max(diff_norm)={diff_norm.max():.1f} at row {np.argmax(diff_norm)}")
    print(f"  max(diff_raw)={diff_raw.max():.1f} at row {np.argmax(diff_raw)}")
    
    # Gradient
    k = np.ones(15)/15
    dsm = np.convolve(diff_norm, k, mode='same')
    reg = dsm[search_start:search_end]
    if np.max(reg) >= 10:
        grads = np.abs(np.diff(reg))
        argmax_row = search_start + np.argmax(grads)
        print(f"  Detection row: {argmax_row}, height={(920-argmax_row)/77.33:.3f}")
    else:
        print(f"  Not detected (max reg={np.max(reg):.1f})")
    print()
