import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- Tunable detection parameters ---
INTENSITY_OFFSET       = 35    # threshold = max(dry_tape_mean - INTENSITY_OFFSET, 30)
MORPH_CLOSE_KERNEL     = (7, 7) # fills holes from foam, glare, tape markings
MORPH_OPEN_KERNEL      = (5, 5) # removes isolated noise and droplet pixels
MIN_WATER_COLUMNS      = 5     # minimum columns with water pixels required
PREVIEW_EVERY_N_FRAMES = 128   # progress log interval

# --- Fallback darkness detection (strict) ---
ADAPTIVE_OFFSET        = 60    # below dry_tape_mean
SUSTAINED_DARK_PIXELS  = 5

video_path = sys.argv[1]
output_path = sys.argv[2]

if not Path("calibration.json").exists():
    print("calibration.json not found. Run calibrate.py first.")
    sys.exit(1)

with open("calibration.json") as f:
    cal = json.load(f)

pixels_per_cm = cal["pixels_per_cm"]
roi = cal["roi"]
baseline_y = cal["baseline_y"]
fps = cal["fps"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]

dry_tape_mean = cal.get("dry_tape_mean")
if dry_tape_mean is None:
    dry_tape_mean = 195
    print("Warning: dry_tape_mean not in calibration.json; using default threshold 195")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Failed to open video: {video_path}")
    sys.exit(1)

ret, frame = cap.read()
if not ret:
    print("Failed to read first frame from video.")
    sys.exit(1)

frame_h, frame_w = frame.shape[:2]
if x + w > frame_w or y + h > frame_h:
    print(f"ERROR: ROI ({x},{y},{w},{h}) exceeds frame dimensions ({frame_w},{frame_h})")
    sys.exit(1)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
results = []
frame_index = 0

clipped_warned = False

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(frame_index / fps, 4)
        roi_frame = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        surface_y = None
        method = None
        wave_height_cm = None

        # --- Primary: intensity threshold + morphology + column-wise median ---
        threshold_intensity = max(dry_tape_mean - INTENSITY_OFFSET, 30)
        _, mask = cv2.threshold(blurred, threshold_intensity, 255, cv2.THRESH_BINARY_INV)

        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

        surface_ys = []
        for col in range(w):
            column = mask[:, col]
            nonzero = np.where(column > 0)[0]
            if len(nonzero) > 0:
                surface_ys.append(nonzero[0])

        if len(surface_ys) >= MIN_WATER_COLUMNS:
            surface_y = int(np.median(surface_ys))
            method = "intensity"
            if surface_y == 0 and not clipped_warned:
                print(f"Warning: surface clipped at ROI top edge, frame {frame_index}")
                clipped_warned = True
            pixel_displacement = baseline_y - surface_y
            wave_height_cm = round(pixel_displacement / pixels_per_cm, 3)
        else:
            # --- Fallback: sustained darkness scan ---
            threshold_dark = dry_tape_mean - ADAPTIVE_OFFSET
            surface_ys_dark = []
            for col in range(w):
                col_vals = blurred[:, col]
                kernel = np.ones(SUSTAINED_DARK_PIXELS, dtype=int)
                convolved = np.convolve((col_vals < threshold_dark).astype(int), kernel, mode='valid')
                matches = np.where(convolved == SUSTAINED_DARK_PIXELS)[0]
                if len(matches) > 0:
                    surface_ys_dark.append(matches[0])
            if len(surface_ys_dark) >= MIN_WATER_COLUMNS:
                surface_y = int(np.median(surface_ys_dark))
                method = "darkness"
                if surface_y == 0 and not clipped_warned:
                    print(f"Warning: surface clipped at ROI top edge, frame {frame_index}")
                    clipped_warned = True
                pixel_displacement = baseline_y - surface_y
                wave_height_cm = round(pixel_displacement / pixels_per_cm, 3)

        results.append({
            'frame_number': frame_index,
            'timestamp_s': timestamp,
            'wave_height_cm': wave_height_cm,
            'detection_method': method
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

plot_frame = df.dropna(subset=['wave_height_cm'])
if len(plot_frame) > 0:
    plt.figure(figsize=(12, 5))
    plt.plot(plot_frame['timestamp_s'], plot_frame['wave_height_cm'], linewidth=0.5)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, label='Baseline')
    plt.title('Wave Height Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Wave Height (cm)')
    plot_path = Path(output_path).parent / "wave_height_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")
else:
    print("No valid detections to plot.")
