import cv2
import numpy as np
import json

with open("calibration.json") as f:
    cal = json.load(f)

import process

roi = cal['roi']

for frame_num in [1, 500, 750, 1000, 1500, 2000, 2500, 2900]:
    cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    
    surface_y, thresh, confidence = process.detect_surface(frame, cal)
    
    baseline_y = cal['baseline_y']
    pixel_displacement = baseline_y - surface_y if surface_y else 0
    wave_height_cm = max(0.0, pixel_displacement) / cal['pixels_per_cm']
    
    print(f"\n=== Frame {frame_num} ===")
    print(f"Surface absolute y: {surface_y}")
    print(f"Confidence: {confidence}")
    print(f"Height: {wave_height_cm:.3f} cm")
    
    if thresh is not None:
        total = thresh.size
        white = np.sum(thresh == 255)
        print(f"Otsu white fraction: {white/total:.1%} ({white}/{total})")
