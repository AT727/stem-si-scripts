# AGENTS.md — Wave Height Analysis Pipeline

Automates wave height measurement from wave tank MP4s (Python + OpenCV),
replacing manual frame-by-frame tracking in Tracker.

**Measured:** vertical water surface displacement at a fixed tape location, in cm,
relative to still-water baseline (zero). Videos are 128 fps × 10+ min = 76,000+
frames. Pipeline replaces O(n) human effort with a one-time ~2 min calibration
then fully automated processing. Read this first for domain context, then SPEC.md
for implementation.

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

---

## Commands

```bash
pip install opencv-python numpy pandas matplotlib
python calibrate.py PhaseII_TestD_0001_c4_01.MP4
python analyze.py PhaseII_TestD_0001_c4_01.MP4 output.csv
```

---

## File Map

```
project/
├── AGENTS.md                        ← domain context
├── SPEC.md                          ← implementation spec
├── opencode.jsonc                   ← opencode config
├── calibrate.py                     ← calibrate.py (to be created)
├── analyze.py                       ← analyze.py (to be created)
└── calibration.json / output.csv / plots  ← runtime artifacts
```

## Superpowers
- Always use using-superpowers when executing any task 
- Always use systematic-debugging when fixing bugs
- Never use finishing-a-development-branch because I do all version control