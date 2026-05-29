import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# --- Tunable detection parameters ---
TAPE_EDGE_COLS    = 60    # Number of columns to use for median profile
TAPE_COL_OFFSET   = 0     # Starting column offset in ROI
DIFF_SMOOTH       = 7     # Boxcar kernel for smoothing temporal diff profile
MIN_DIFF_CONFIDENCE = 6   # Minimum peak temporal diff to confirm wave motion; else treat as baseline
SEARCH_START_FRAC = 0.25  # Fraction of ROI height to start water surface search
SEARCH_END_FRAC   = 0.95  # Fraction of ROI height to end search

PREVIEW_EVERY_N_FRAMES = 128

video_path = sys.argv[1]
output_path = sys.argv[2]

if not Path("calibration.json").exists():
    print("calibration.json not found. Run calibrate.py first.")
    sys.exit(1)

with open("calibration.json") as f:
    cal = json.load(f)

pixels_per_cm = cal["pixels_per_cm"]
roi = cal["roi"]
fps = cal["fps"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

col_start = min(TAPE_COL_OFFSET, w - 1)
col_end = min(col_start + TAPE_EDGE_COLS, w)
col_slice = slice(col_start, col_end)
search_start = int(h * SEARCH_START_FRAC)
search_end = int(h * SEARCH_END_FRAC)
baseline_y = search_end

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("Failed to read first frame.")
    sys.exit(1)

frame_h, frame_w = frame.shape[:2]
if x + w > frame_w or y + h > frame_h:
    print(f"ERROR: ROI ({x},{y},{w},{h}) exceeds frame dimensions ({frame_w},{frame_h})")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
results = []

def detect_water_surface(roi_gray, prev_roi_gray, col_slice, search_start, search_end):
    curr = roi_gray[:, col_slice].astype(np.float32)
    prev = prev_roi_gray[:, col_slice].astype(np.float32)
    
    if curr.shape != prev.shape:
        prev = cv2.resize(prev, (curr.shape[1], curr.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    diff = np.median(np.abs(curr - prev), axis=1)
    
    k = np.ones(DIFF_SMOOTH) / DIFF_SMOOTH
    diff_sm = np.convolve(diff, k, mode='same')
    
    reg = diff_sm[search_start:search_end]
    
    if np.max(reg) < MIN_DIFF_CONFIDENCE:
        return search_end
    
    return search_start + np.argmax(reg)

# --- Process all frames using frame-differencing ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
prev_roi_gray = None
frame_count = 0

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = round(frame_count / fps, 4)
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        if prev_roi_gray is None:
            surface_y = search_end
            prev_roi_gray = roi_gray
        else:
            surface_y = detect_water_surface(roi_gray, prev_roi_gray, col_slice,
                                              search_start, search_end)
            prev_roi_gray = roi_gray

        pixel_displacement = baseline_y - surface_y
        wave_height_cm = round(pixel_displacement / pixels_per_cm, 3)
        results.append({
            'frame_number': frame_count,
            'timestamp_s': timestamp,
            'wave_height_cm': wave_height_cm
        })
        frame_count += 1
        if frame_count % PREVIEW_EVERY_N_FRAMES == 0:
            pct = 100 * frame_count / total_frames
            print(f"Processed frame {frame_count} / {total_frames} ({pct:.1f}%)")
    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        frame_count += 1
        continue

cap.release()

df = pd.DataFrame(results)
df.to_csv(output_path, index=False)

print(f"\nTotal frames processed: {frame_count}")
none_count = df['wave_height_cm'].isna().sum()
print(f"Detection failures: {none_count}")

valid = df['wave_height_cm'].dropna()
if len(valid) > 0:
    print(f"Min wave height: {valid.min():.3f} cm")
    print(f"Max wave height: {valid.max():.3f} cm")
    print(f"Mean wave height: {valid.mean():.3f} cm")
else:
    print("No valid detections.")

import matplotlib.pyplot as plt
plot_frame = df.dropna(subset=['wave_height_cm'])
plt.figure(figsize=(12, 5))
plt.plot(plot_frame['timestamp_s'], plot_frame['wave_height_cm'], linewidth=0.5)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, label='Baseline')
plt.title('Wave Height Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Wave Height (cm)')
plot_path = Path(output_path).parent / "wave_height_plot.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
