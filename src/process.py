import cv2
import numpy as np
import json
from sklearn.linear_model import RANSACRegressor

def load_cal():
    with open("calibration.json", "r") as f:
        return json.load(f)

def detect_surface(frame, cal):
    # Masking
    roi = cal['roi']
    img = frame[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    
    # Mask ceiling
    ceiling_y_local = cal['wave_ceiling_y'] - roi['y']
    img[:ceiling_y_local, :] = 255
    
    # Threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Contour filtering
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= 8:
            candidates.append(y + roi['y'])
    
    if not candidates: return None
    return min(candidates)

def process_video(video_path):
    cal = load_cal()
    cap = cv2.VideoCapture(video_path)
    
    # Pass 1: Baseline
    baseline_y = cal['baseline_y']
    
    # Pass 2: Detection
    results = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        surface_y = detect_surface(frame, cal)
        if surface_y is None: surface_y = baseline_y
        
        # RANSAC (Simplified aggregator)
        # Note: True RANSAC would be applied over multiple columns/frames here.
        
        pixel_displacement = baseline_y - surface_y
        wave_height_cm = round(max(0.0, pixel_displacement) / cal['pixels_per_cm'], 3)
        results.append(wave_height_cm)
        
    return results
