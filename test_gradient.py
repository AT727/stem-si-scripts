import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS = 60

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
search_start = int(0.25 * h)
search_end = int(0.95 * h)

for fn in [0, 100, 150, 300, 400, 500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    sub = rg[:, :TAPE_EDGE_COLS].astype(np.float32)
    profile = np.median(sub, axis=1)
    
    k = np.ones(15)/15
    smooth = np.convolve(profile, k, mode='same')
    
    grad = -np.diff(smooth)  # positive = downward gradient
    reg = grad[search_start:search_end]
    
    if np.max(reg) < 1.0:
        surface_row = search_end
    else:
        surface_row = search_start + np.argmax(reg)
    
    height = (search_end - surface_row) / 77.33
    print(f"Frame {fn}: profile range [{profile.min():.0f}, {profile.max():.0f}]")
    print(f"  max gradient={np.max(reg):.2f}, surface_row={surface_row}, height={height:.3f}")
