import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
print(f"ROI: x={x}, y={y}, w={w}, h={h}")

cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
ret, frame = cap.read()
rg = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

# Check per-column brightness at Sharpie row (317) and nearby rows
print(f"\nPer-column brightness at row 317 (Sharpie area):")
print("col: ", end="")
for c in range(w):
    print(f"{c:4d}", end=" ")
print()
print("val: ", end="")
for c in range(w):
    print(f"{rg[317,c]:4d}", end=" ")
print()

print(f"\nPer-column brightness at row 315 (above Sharpie):")
for c in range(w):
    print(f"{rg[315,c]:4d}", end=" ")
print()

print(f"\nPer-column brightness at row 320 (below Sharpie):")
for c in range(w):
    print(f"{rg[320,c]:4d}", end=" ")
print()

# Also check overall profile per column group
left60 = np.median(rg[:, :60], axis=1)
right45 = np.median(rg[:, 60:], axis=1)
mid_cols = np.median(rg[:, 30:60], axis=1)

print(f"\nSharpie row in left60: {left60[317]:.1f}")
print(f"Sharpie row in right45: {right45[317]:.1f}")
print(f"Sharpie row in mid_cols: {mid_cols[317]:.1f}")

# Check the gradient strength of Sharpie in left60 vs right45
k = np.ones(15)/15
print(f"\nLeft60 gradient at row 317: {-np.diff(np.convolve(left60, k, mode='same'))[317]:.2f}")
print(f"Right45 gradient at row 317: {-np.diff(np.convolve(right45, k, mode='same'))[317]:.2f}")
