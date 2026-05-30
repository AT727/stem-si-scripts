import cv2
import numpy as np
import json

with open("calibration.json") as f:
    cal = json.load(f)

roi = cal['roi']
fps = cal['fps']
cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")

for frame_num in [1, 500, 1000, 1500, 2000, 2500]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret: continue
    
    roi_view = frame[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']].copy()
    
    ceiling_y_local = cal['wave_ceiling_y'] - roi['y']
    
    gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY)
    masked = gray.copy()
    masked[:ceiling_y_local, :] = 255
    
    print(f"\n=== Frame {frame_num} ===")
    
    # Image statistics
    below = masked[ceiling_y_local:, :]
    print(f"Below ceiling: min={below.min()}, max={below.max()}, mean={below.mean():.1f}, std={below.std():.1f}")
    
    # Percentiles
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(below, p)
        print(f"  {p}th percentile: {val:.0f}")
    
    # Otsu threshold without blur
    _, thresh_raw = cv2.threshold(below, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_raw = cv2.threshold(below, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    print(f"Otsu threshold (no blur): {otsu_raw}")
    
    # Otsu threshold with blur
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    _, thresh_blur = cv2.threshold(blurred[ceiling_y_local:, :], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu_blur = cv2.threshold(blurred[ceiling_y_local:, :], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
    print(f"Otsu threshold (with blur): {otsu_blur}")
    
    # Check if ceiling mask creates issues
    # What fraction of below-ceiling pixels are white in the thresholded image?
    white_frac_raw = np.sum(thresh_raw == 255) / thresh_raw.size
    white_frac_blur = np.sum(thresh_blur == 255) / thresh_blur.size
    print(f"White fraction (no blur): {white_frac_raw:.1%}")
    print(f"White fraction (with blur): {white_frac_blur:.1%}")
    
    # Try on the 50th row from ceiling: what does the grayscale histogram look like?
    sample_row = below[min(100, below.shape[0]-1), :]
    print(f"Row at y=100: min={sample_row.min()}, max={sample_row.max()}, mean={sample_row.mean():.1f}")

cap.release()
