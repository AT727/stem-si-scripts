# Water Level Extraction Pipeline

This system extracts frame-by-frame water level measurements from high-speed video of a wavetank. It is designed to be robust against common issues in this environment, specifically:
- **Sharpie-Immune:** Explicitly designed to ignore hand-drawn marker lines on reference tapes.
- **Bimodal Histogram Normalization:** Handles complex backgrounds (like ceiling infrastructure) using mask-based normalization.
- **Robust Aggregation:** Uses RANSAC-based surface fitting to reject foam, droplets, and surface noise.

---

## Prerequisites

- Python 3
- OpenCV (`cv2`)
- NumPy
- Scikit-learn

Install dependencies:
```bash
pip install opencv-python numpy scikit-learn
```

---

## Workflow

### 1. Calibration (`calibrate.py`)
This step defines the relationship between pixel coordinates in your video and real-world centimeters.

Run the calibration tool on your video file:
```bash
python src/calibrate.py <video_path> <dist_cm>
```
Example:
```bash
python src/calibrate.py PhaseII_TestD_0003_c4_01.MP4 8.2
```

**Interactive Steps:**
1. **Define ROI:** A window will open. Draw a box around the reference tape column using your mouse, then press `ENTER` or `SPACE`.
2. **Click Reference Points:** In the zoomed-in window, click three points in order:
   - **Click 1:** Still-water baseline (this is your **zero reference**).
   - **Click 2:** A known reference point some distance above Click 1.
   - **Click 3:** The wave ceiling (the highest point you expect waves to reach).
3. **Press any key** when done clicking.
4. **Validation:** The script will check if you have sufficient "headroom" above the baseline to capture waves.

The results are saved to `calibration.json` in the root directory.

### 2. Processing (`process.py`)
This step runs the detection pipeline using your generated `calibration.json`.

**Note:** Ensure `src/process.py` is configured to output your results as a CSV.

To process a video, use the `process_video` function:
```python
from src.process import process_video
results = process_video("your_video_file.MP4")

# Export to CSV
import csv
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "wave_height_cm"])
    for i, h in enumerate(results):
        writer.writerow([i, h])
```

---

## Design Logic

- **Ceiling Masking:** The pipeline masks the ceiling infrastructure to pure white before thresholding, ensuring the thresholding algorithm only sees the contrast between the silver tape and the dark water.
- **Contour Filtering:** The detector filters out contours smaller than 8 pixels in height. Since the Sharpie line is <5px, it is effectively ignored.
- **RANSAC Fitting:** Instead of measuring a single pixel, the pipeline fits a line through all detected water surface points in the frame using RANSAC. This provides a robust measurement that rejects outliers like foam or droplets.
