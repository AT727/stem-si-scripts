# Implementation Plan: Tape-Transition Detection

Replace intensity-threshold-on-water detection with tape dry-to-wet transition detection.

## Files to modify

- `calibrate.py` — add dry-tape reference sample step
- `analyze.py` — new hybrid gradient/darkness detection pipeline

---

## calibrate.py changes

**After** the two calibration clicks (line ~90, before saving), insert:

1. Display ROI with instruction: "Click a point on the dry tape above the waterline"
2. Capture one click via existing mouse_callback
3. Sample 15x15 patch around click, compute `np.mean(gray[patch])`
4. Save as `dry_tape_mean` in `calibration.json`
5. Draw patch on confirmation overlay

If user presses a key without clicking: warn but save without the field.

---

## calibration.json schema (new field)

```json
{
  "pixels_per_cm": 12.4,
  "roi": { "x": 310, "y": 80, "w": 60, "h": 420 },
  "baseline_y": 210,
  "fps": 128.0,
  "dry_tape_mean": 195.3
}
```

---

## analyze.py — New constants

```python
# --- Gradient detection (PRIMARY) ---
MIN_GRADIENT_MAGNITUDE = 30
SOBEL_KSIZE            = 3

# --- Darkness detection (FALLBACK) ---
ADAPTIVE_OFFSET        = 60
SUSTAINED_DARK_PIXELS  = 5

# --- Shared ---
MIN_WATER_COLUMNS      = 5
PREVIEW_EVERY_N_FRAMES = 128
```

---

## analyze.py — Per-frame detection logic

```
for each frame:
    1. Crop to ROI, grayscale, blur (unchanged)

    2. GRADIENT METHOD (primary):
       sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
       valid_cols = []
       for col in range(roi_w):
           response = sobel[:, col]
           min_idx = np.argmin(response)
           if response[min_idx] < -MIN_GRADIENT_MAGNITUDE:
               valid_cols.append(min_idx)
       
       if len(valid_cols) >= MIN_WATER_COLUMNS:
           surface_y = int(np.median(valid_cols))
           method = "gradient"
       else:
           3. DARKNESS METHOD (fallback):
              threshold = dry_tape_mean - ADAPTIVE_OFFSET
              valid_cols = []
              for col in range(roi_w):
                  col_vals = blurred[:, col]
                  for y in range(roi_h - SUSTAINED_DARK_PIXELS):
                      if all(col_vals[y+k] < threshold for k in range(SUSTAINED_DARK_PIXELS)):
                          valid_cols.append(y)
                          break
              
              if len(valid_cols) >= MIN_WATER_COLUMNS:
                  surface_y = int(np.median(valid_cols))
                  method = "darkness"
              else:
                  wave_height_cm = None
                  method = None

    4. Compute displacement from baseline_y (unchanged)
    5. Append result with detection_method
```

---

## CSV schema (new column)

| Column | Type | Description |
|---|---|---|
| `frame_number` | int | 0-indexed |
| `timestamp_s` | float | `frame_number / fps` |
| `wave_height_cm` | float or None | Displacement above baseline |
| `detection_method` | str or None | `"gradient"`, `"darkness"`, or None |

---

## Error handling

| Condition | Behavior |
|---|---|
| `calibration.json` missing `dry_tape_mean` | Use default threshold value `195 - ADAPTIVE_OFFSET`, print warning |
| Gradient finds too few columns | Silently fall through to darkness method |
| Both methods fail | `wave_height_cm = None`, `detection_method = None` |
| `surface_y == 0` | Same warning as before |

---

## Verification

1. `python calibrate.py <video>` — confirm dry-tape click works, `dry_tape_mean` in JSON
2. `python analyze.py <video> output.csv` — confirm CSV has `detection_method` column
3. Spot-check: most wave frames should use `"gradient"` method
