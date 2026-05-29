# SPEC.md — Implementation Specification

Read AGENTS.md first for domain context before reading this file.

---

## Scripts to Build

### 1. calibrate.py
One-time setup. Establishes pixel-to-cm conversion and saves calibration.json.

### 2. analyze.py
Batch processing. Reads calibration.json, processes every frame, outputs CSV.

---

## Usage

```
python calibrate.py wave.mp4
python analyze.py wave.mp4 output.csv
```

---

## calibrate.py — Full Spec

### Step 1 — Load first frame
- Open the MP4 passed as `sys.argv[1]` with `cv2.VideoCapture`
- Read only the first frame with `cap.read()`
- Print `cap.get(cv2.CAP_PROP_FPS)` and `cap.get(cv2.CAP_PROP_FRAME_COUNT)` so
  the user can verify frame rate and total frame count
- If FPS is 0 or < 1, prompt the user to enter it manually via `input()`

### Step 2 — ROI selection
- Display the first frame with `cv2.imshow`
- Print instructions: draw a rectangle tightly around the measuring tape column
  only. Explicitly warn: exclude the right side of the frame (blue foam board)
- Use `cv2.selectROI` to capture `(x, y, w, h)` from the user
- If `w > 200`, print a warning that the ROI may be too wide and include
  non-tape regions

### Step 3 — Baseline and reference point selection
- Close the ROI window
- Display the cropped ROI region in a new window
- Print instructions: click two points on the measuring tape
    - Click 1: the zero/still-water baseline mark
    - Click 2: a known reference point N cm above it
- Capture both clicks with `cv2.setMouseCallback`
- After both clicks are captured, prompt via `input()` for the real-world
  distance in cm between them

### Step 4 — Compute and save calibration
- `pixels_per_cm = abs(point1_y - point2_y) / entered_cm`
- If `abs(point1_y - point2_y) < 5`, print an error and re-prompt — points are
  too close to produce an accurate ratio
- `baseline_y` = the y-coordinate of the zero/still-water click **in
  ROI-relative coordinates**. Explicitly subtract the ROI origin `y` from the
  raw clicked y-coordinate before saving. This is the most common source of
  silent measurement error.
- Save `calibration.json` with fields:
  ```json
  {
    "pixels_per_cm": 12.4,
    "roi": { "x": 310, "y": 80, "w": 60, "h": 420 },
    "baseline_y": 210,
    "fps": 128.0
  }
  ```

### Step 5 — Confirmation
- Print the computed `pixels_per_cm`
- Overlay the two clicked points and a horizontal baseline line on the ROI
- Display briefly with `cv2.imshow` so the user can visually verify accuracy
- Print `"Saved calibration.json"` and read the file back to confirm it is
  valid JSON before exiting

---

## analyze.py — Full Spec

### Constants block
Define at the top of the file. All tunable values are named constants with
comments explaining when to adjust them.

```python
# --- Tunable detection parameters ---
INTENSITY_THRESHOLD    = 80    # pixels below this brightness (0–255) are
                               # classified as water. Raise if background noise
                               # triggers false positives. Lower if wave water
                               # is not being detected.
MORPH_CLOSE_KERNEL     = (7, 7) # fills holes from foam, glare, tape markings
MORPH_OPEN_KERNEL      = (5, 5) # removes isolated noise and droplet pixels
MIN_WATER_COLUMNS      = 5     # minimum columns with water pixels required to
                               # count a frame as a valid detection. Prevents
                               # isolated droplets from producing a reading.
PREVIEW_EVERY_N_FRAMES = 128   # progress log interval
```

### Step 1 — Setup
- Accept `sys.argv[1]` (video path) and `sys.argv[2]` (output CSV path)
- Load `calibration.json` from the working directory. If it does not exist,
  print `"calibration.json not found. Run calibrate.py first."` and
  `sys.exit(1)`
- Extract `pixels_per_cm`, `roi` (x, y, w, h), `baseline_y`, `fps`
- Open video with `cv2.VideoCapture`. If it fails to open, print the path and
  exit cleanly
- Read `total_frames` with `cap.get(cv2.CAP_PROP_FRAME_COUNT)`
- Initialize `results = []` and `frame_index = 0`

### Step 2 — Per-frame loop

Use `while cap.isOpened()`. Wrap the entire body in `try/except` — log the
frame index on any exception and continue. Never crash the full run on a single
bad frame.

For each frame:

**Read**
```python
ret, frame = cap.read()
if not ret:
    break
timestamp = round(frame_index / fps, 4)
```

**Crop to ROI**
```python
roi_frame = frame[y:y+h, x:x+w]
```

**Convert to grayscale and blur**
```python
gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
```

**Intensity threshold — detect dark water mass**
```python
_, mask = cv2.threshold(blurred, INTENSITY_THRESHOLD,
                        255, cv2.THRESH_BINARY_INV)
# THRESH_BINARY_INV: pixels BELOW threshold → 255 (water)
#                    pixels ABOVE threshold → 0   (air)
```

**Morphological cleanup**
```python
close_k = cv2.getStructuringElement(
    cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)
open_k  = cv2.getStructuringElement(
    cv2.MORPH_RECT, MORPH_OPEN_KERNEL)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)
```

**Surface detection — column-by-column median**
```python
surface_ys = []
for col in range(mask.shape[1]):
    column = mask[:, col]
    nonzero = np.where(column > 0)[0]
    if len(nonzero) > 0:
        surface_ys.append(nonzero[0])  # topmost water pixel in column

if len(surface_ys) < MIN_WATER_COLUMNS:
    wave_height_cm = None  # not enough columns — detection failure
else:
    surface_y = int(np.median(surface_ys))
    if surface_y == 0:
        # Surface at top of ROI — may indicate ROI needs to be extended upward
        # Log a warning every time this happens
        print(f"Warning: surface clipped at ROI top edge, frame {frame_index}")
    pixel_displacement = baseline_y - surface_y  # positive = above baseline
    wave_height_cm = round(pixel_displacement / pixels_per_cm, 3)
```

**Store result**
```python
results.append({
    'frame_number': frame_index,
    'timestamp_s':  timestamp,
    'wave_height_cm': wave_height_cm
})
frame_index += 1
```

**Progress log**
```python
if frame_index % PREVIEW_EVERY_N_FRAMES == 0:
    print(f"Processed frame {frame_index} / {total_frames} "
          f"({100 * frame_index / total_frames:.1f}%)")
```

### Step 3 — Output

**CSV**
```python
df = pd.DataFrame(results)
df.to_csv(sys.argv[2], index=False)
```

**Summary print**
- Total frames processed
- Number of frames where detection failed (None values)
- Min / max / mean wave height (excluding None)

**Diagnostic plot**
```python
# Save alongside the CSV, not always in the working directory
from pathlib import Path
plot_path = Path(sys.argv[2]).parent / "wave_height_plot.png"
```
- Drop None rows before plotting
- x-axis: `timestamp_s`, y-axis: `wave_height_cm`
- Title: `"Wave Height Over Time"`
- x-label: `"Time (s)"`, y-label: `"Wave Height (cm)"`
- Add a horizontal dashed line at y=0 (baseline reference)
- Save as PNG, do not call `plt.show()`

---

## calibration.json Schema

```json
{
  "pixels_per_cm": 12.4,
  "roi": { "x": 310, "y": 80, "w": 60, "h": 420 },
  "baseline_y": 210,
  "fps": 128.0
}
```

| Field | Description |
|---|---|
| `pixels_per_cm` | Spatial conversion factor for this video |
| `roi` | Bounding box around the tape in full-frame pixel coordinates |
| `baseline_y` | Still-water y-coordinate in **ROI-relative** coordinates |
| `fps` | Frames per second |

---

## Output CSV Schema

| Column | Type | Description |
|---|---|---|
| `frame_number` | int | 0-indexed |
| `timestamp_s` | float | `frame_number / fps`, 4 decimal places |
| `wave_height_cm` | float or None | Displacement above baseline; None = detection failure |

`None` values must be preserved as empty cells (NaN) in the CSV. Do not replace
with 0.

---

## Error Handling Requirements

| Condition | Required behavior |
|---|---|
| `calibration.json` missing | Print message, `sys.exit(1)` |
| Video file won't open | Print path, exit cleanly |
| `fps` from codec is 0 or < 1 | Prompt user to enter fps manually |
| Single bad frame in loop | Log frame index, continue — never crash full run |
| Fewer than `MIN_WATER_COLUMNS` detect water | Set `wave_height_cm = None`, log as failure |
| `surface_y == 0` | Print warning — ROI may need to extend upward |
| Calibration clicks too close (`< 5px`) | Print error, re-prompt for clicks |

---

## Environment Requirements

- **calibrate.py**: requires a windowed (GUI) environment — uses `cv2.imshow`,
  `cv2.selectROI`, `cv2.setMouseCallback`. Will not work headless.
- **analyze.py**: fully headless-safe — outputs files only, no display.

---

## What NOT to Do

- Do not use `cv2.findContours` as primary surface detection — fragmented masks
  from turbulence make contour hierarchies unreliable
- Do not hardcode pixel values — everything spatial must come from
  `calibration.json`
- Do not silently replace `None` with `0` in the CSV
- Do not add `cv2.imshow` preview inside the `analyze.py` frame loop
- Do not draw the ROI to include the right side of the frame (blue foam board)
- Do not subtract the ROI origin twice — `baseline_y` is already ROI-relative
  after calibration; do not re-subtract during analysis

---

## Known Edge Cases

| # | Case | Mitigation |
|---|---|---|
| 1 | `baseline_y` saved as full-frame coordinate | Explicitly subtract ROI `y` in calibrate.py before saving |
| 2 | `fps` returns 0 from codec | Guard + manual input prompt |
| 3 | Still water nearly invisible | Baseline is user-clicked on tape mark, not auto-detected |
| 4 | Wave fills entire ROI (`surface_y == 0`) | Print warning; user should extend ROI upward |
| 5 | Dense mist above surface | Column-median ignores isolated droplet columns; increase `MORPH_OPEN_KERNEL` if needed |
| 6 | Calibration clicks too close | Guard: `abs(y1 - y2) < 5px` → error + re-prompt |
| 7 | Tape tick marks as dark pixels | Consistent across frames, won't shift surface detection |
| 8 | 76,000+ frames at 128fps | Tight ROI + grayscale threshold keeps per-frame cost low |