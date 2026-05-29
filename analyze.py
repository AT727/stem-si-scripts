import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# --- Tunable detection parameters ---
TAPE_EDGE_COLS    = 60    # Number of columns to use for median profile
TAPE_COL_OFFSET   = 0     # Starting column offset in ROI
BASELINE_FRAMES   = 100   # Number of initial frames for brightness baseline
DIFF_SMOOTH       = 101   # Large boxcar kernel for smoothing diff profile (suppresses tick marks)
MIN_DIFF_CONFIDENCE = 4   # Minimum peak abs difference to confirm detection; else treat as baseline
SEARCH_START_FRAC = 0.25  # Fraction of ROI height to start water surface search
SEARCH_END_FRAC   = 0.95  # Fraction of ROI height to end search
AIR_REF_ROWS      = 30    # Top rows used as air brightness reference

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
frame_index = 0

def detect_water_surface(roi_gray, baseline, col_slice, search_start, search_end, air_ref):
    sub = roi_gray[:, col_slice].astype(np.float32)
    # Air brightness normalization (corrects camera auto-exposure)
    air_frame = np.median(sub[:AIR_REF_ROWS])
    if air_ref > 0 and air_frame > 0:
        scale = air_ref / air_frame
        sub = sub * scale
    sub = np.clip(sub, 0, 255)
    if baseline.shape != sub.shape:
        bl = cv2.resize(baseline, (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        bl = baseline

    diff = np.median(np.abs(sub - bl), axis=1)

    k = np.ones(DIFF_SMOOTH) / DIFF_SMOOTH
    diff_sm = np.convolve(diff, k, mode='same')

    # Subtract median diff in air region to cancel residual exposure drift
    air_offset = np.median(diff_sm[:AIR_REF_ROWS * 4])
    diff_norm = np.maximum(diff_sm - air_offset, 0)

    reg = diff_norm[search_start:search_end]

    if np.max(reg) < MIN_DIFF_CONFIDENCE:
        return search_end

    gradients = np.diff(reg)
    if len(gradients) == 0:
        return search_end
    return search_start + np.argmax(gradients)

# --- First pass: accumulate baseline frames ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
baseline_frames_list = []
baseline_count = 0
while baseline_count < BASELINE_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    baseline_frames_list.append(roi_gray)
    results.append({
        'frame_number': baseline_count,
        'timestamp_s': round(baseline_count / fps, 4),
        'wave_height_cm': 0.0
    })
    baseline_count += 1
    frame_index += 1

n_baseline = len(baseline_frames_list)
if n_baseline == 0:
    print("Failed to read any frames for baseline.")
    sys.exit(1)

baseline_stack = np.stack(baseline_frames_list, axis=0)[:, :, col_slice]
baseline_profile = np.median(baseline_stack, axis=0).astype(np.float32)
baseline_frames_list = None
baseline_y = float(search_end)
air_ref = float(np.median(baseline_profile[:AIR_REF_ROWS]))
print(f"Baseline frames: {n_baseline}")
print(f"Baseline (still water row): {baseline_y:.0f} ROI-relative, "
      f"pixels_per_cm: {pixels_per_cm}")
print(f"Air brightness reference: {air_ref:.1f}")

# --- Second pass: process remaining frames ---
while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = round(frame_index / fps, 4)
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        surface_y = detect_water_surface(roi_gray, baseline_profile, col_slice,
                                          search_start, search_end, air_ref)
        pixel_displacement = baseline_y - surface_y
        wave_height_cm = round(pixel_displacement / pixels_per_cm, 3)
        if surface_y == 0:
            print(f"Warning: surface clipped at ROI top edge, frame {frame_index}")
        results.append({
            'frame_number': frame_index,
            'timestamp_s': timestamp,
            'wave_height_cm': wave_height_cm
        })
        frame_index += 1
        if frame_index % PREVIEW_EVERY_N_FRAMES == 0:
            pct = 100 * frame_index / total_frames
            print(f"Processed frame {frame_index} / {total_frames} ({pct:.1f}%)")
    except Exception as e:
        print(f"Error processing frame {frame_index}: {e}")
        frame_index += 1
        continue

cap.release()

df = pd.DataFrame(results)
df.to_csv(output_path, index=False)

print(f"\nTotal frames processed: {frame_index}")
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
