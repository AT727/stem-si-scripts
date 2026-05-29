import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, f0 = cap.read()
f0_gray = cv2.cvtColor(f0[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

cap.set(cv2.CAP_PROP_POS_FRAMES, 400)
ret, f400 = cap.read()
f400_gray = cv2.cvtColor(f400[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

# Per-column diff at the water surface row (430)
print("Per-column |f0-f400| at row 430 (water surface):")
for c in range(30):
    print(f"  col {c:3d}: {abs(int(f0_gray[430,c])-int(f400_gray[430,c])):3d} (f0={int(f0_gray[430,c]):3d}, f400={int(f400_gray[430,c]):3d})")

# Per-column diff at row 100 (well above surface)
print("\nPer-column |f0-f400| at row 100 (air):")
for c in range(30):
    d = abs(int(f0_gray[100,c])-int(f400_gray[100,c]))
    if d > 5:
        print(f"  col {c:3d}: {d:3d} (f0={int(f0_gray[100,c]):3d}, f400={int(f400_gray[100,c]):3d})")
