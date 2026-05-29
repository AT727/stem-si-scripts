import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
ret, frame = cap.read()
rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

# Check which columns look like tape (bright silver) vs background
# by looking at a row in the middle of the ROI (row 500)
print("Columns 0-104 at row 500 (mid-ROI):")
print("col: ", end="")
for c in range(0, 105, 5):
    print(f"{c:5d}", end="")
print()
print("val: ", end="")
for c in range(0, 105, 5):
    print(f"{rg[500,c]:5d}", end="")
print()

# Also check temporal stability: look at columns in the still-water region (row 500) across baseline frames
print("\nPer-column temporal std over first 100 frames at row 500:")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
vals = np.zeros((100, w))
for fn in range(100):
    ret, frame = cap.read()
    rg2 = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    vals[fn] = rg2[500]
stds = np.std(vals, axis=0)
print("col: ", end="")
for c in range(0, 105, 5):
    print(f"{c:5d}", end="")
print()
print("std: ", end="")
for c in range(0, 105, 5):
    print(f"{stds[c]:5.1f}", end="")
print()

# Check which columns change most when water rises
# Frame 0 (still water, no surface visible above bottom) vs frame 400 (wave at row 430)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, f0 = cap.read()
f0_gray = cv2.cvtColor(f0[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

cap.set(cv2.CAP_PROP_POS_FRAMES, 400)
ret, f400 = cap.read()
f400_gray = cv2.cvtColor(f400[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

# Compute absolute difference per column at row 400 (just below water surface)
print(f"\nPer-column |f0-f400| at row 430 (water surface region):")
print("col: ", end="")
for c in range(0, 105, 5):
    print(f"{c:5d}", end="")
print()
print("val: ", end="")
for c in range(0, 105, 5):
    print(f"{abs(int(f0_gray[430,c])-int(f400_gray[430,c])):5d}", end="")
print()

# Per-column |f0-f400| at row 500 (water region, far below surface):
print(f"\nPer-column |f0-f400| at row 500 (submerged):")
print("col: ", end="")
for c in range(0, 105, 5):
    print(f"{c:5d}", end="")
print()
print("val: ", end="")
for c in range(0, 105, 5):
    print(f"{abs(int(f0_gray[500,c])-int(f400_gray[500,c])):5d}", end="")
print()
