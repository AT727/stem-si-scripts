# AGENTS.md — Wave Height Analysis Pipeline

Automates wave height measurement from wave tank MP4s (Python + OpenCV),
replacing manual frame-by-frame tracking in Tracker.

**Measured:** vertical water surface displacement at a fixed tape location, in cm,
relative to still-water baseline (zero). Videos are ~120 fps × 10+ min = 72,000+
frames. Pipeline replaces O(n) human effort with a one-time ~2 min calibration
then fully automated processing.

---

## Video Domain (ground truth from actual footage)

- **Camera:** Stationary, straight-on, backlit through glass tank. Lab setting
  with wet concrete floor, columns, windows on far right.
- **Tape:** Silver metallic body, black tick marks + numerals. Vertical,
  center-right of frame. Visible range ~35–50 cm. Submerged portion disappears.
- **Water is NOT blue.** It is dark brown/gray turbid water — no usable hue
  signal. Detection must rely on intensity (darkness) of the water mass.
  - *Still water:* thin reflective sheen near frame bottom, nearly invisible.
  - *Wave water:* large well-defined dark mass occupying lower ~35% of frame.
    Strong contrast between dark water and lighter air above.
- **Surface is jagged/turbulent.** Foam, spray, and mist above the true surface.
  Single-pixel reads land on droplets. **Column-wise median aggregation is
  mandatory.**
- **Right ~20–25% of frame:** bright blue foam/insulation board (false positive
  hazard for intensity detection). ROI must exclude this area entirely.
- **Airborne mist:** scattered gray pixels above surface — must not be mistaken
  for water.
- **Sharpie Mark** dark black horizontal line at 1100 mm part, to the left of measuring tape.

---

## Detection Algorithm

analyze.py uses an **absolute-difference + gradient** approach (not intensity thresholding):

1. **Baseline:** median brightness profile from first 100 frames (still water)
2. **Per-frame diff:** `median(|frame_ROI_gray_left_60px − baseline_profile|)` → 1D array of absolute brightness change per row (high where water surface differs from baseline, regardless of direction)
3. **Smooth:** boxcar kernel (15px) to suppress turbulence noise
4. **Surface find:** maximum gradient of smoothed profile — the sharpest transition top-to-bottom marks the wave surface (works for both crests and troughs)
5. **Height:** `pixel_displacement / pixels_per_cm`, displacement = baseline row − surface row (negative = trough below baseline)

Column-wise median over leftmost 60px of ROI (`TAPE_EDGE_COLS`) rejects foam/mist outliers.

## Parameters

| Constant | Default | Effect |
|----------|---------|--------|
| `TAPE_EDGE_COLS` | 60 | Leftmost ROI columns for median aggregation |
| `BASELINE_FRAMES` | 100 | Frames averaged for still-water brightness profile |
| `DIFF_SMOOTH` | 15 | Smoothing kernel (px) — higher = less turbulence noise |
| `MIN_DIFF_CONFIDENCE` | 4 | Minimum peak abs diff to confirm detection; else treat as baseline |
| `SEARCH_START_FRAC` | 0.50 | Start search as fraction of ROI height (avoids mist) |
| `SEARCH_END_FRAC` | 0.95 | End search as fraction of ROI height |

## Gotchas

- **ROI must be tight on tape column.** Existing `calibration.json` has `w=1131` (likely includes surrounding area). Re-calibrate with ~60px wide ROI.
- **calibration `baseline_y` is for reference only.** calibrate.py saves the user-clicked baseline (ROI-relative), but analyze.py ignores it — it computes its own algorithmic baseline from `SEARCH_END_FRAC` (`h * 0.95`).
- **Diff algo yields 0 displacement on still water** — correct; crests produce positive height, troughs produce negative height.
- **Bright blue foam board on right ~25% of frame** — ROI must exclude this area (false positive for darkness detection).

## Workflow

1. `python calibrate.py video.mp4` — GUI calibration (requires display)
2. `python analyze.py video.mp4 output.csv` — batch (headless-safe)
3. Inspect `output.csv` and `wave_height_plot.png`

---

## Commands

```bash
pip install opencv-python numpy pandas matplotlib
python calibrate.py PhaseII_TestD_0003_c4_01.MP4
python analyze.py PhaseII_TestD_0003_c4_01.MP4 output.csv
```

---

## File Map

```
AGENTS.md              ← this file — domain context + pipeline docs
calibrate.py           ← GUI calibration: select ROI, click 2 tape points → calibration.json
analyze.py             ← batch process frames → output.csv + wave_height_plot.png
calibration.json       ← created by calibrate, consumed by analyze
output.csv             ← created by analyze — columns: frame_number, timestamp_s, wave_height_cm
wave_height_plot.png
PhaseII_TestD_*.MP4    ← test videos
```

## Superpowers
- Always use superpowers for any task or bug fixing
- Never use finishing-a-development-branch because I do all version control