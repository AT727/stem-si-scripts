---
name: wave-height-analysis
description: >-
  Automates wave height measurement from wave tank MP4s using the project's
  calibrate.py and analyze.py scripts. Use this skill whenever the user mentions:
  processing wave tank videos, running analyze.py or calibrate.py, measuring
  water surface displacement, wave height analysis, benchmarking wave detection
  accuracy, or wanting to improve detection via RMSE/NRMSE metrics. Also use
  when the user wants to tune detection parameters (TAPE_EDGE_COLS, DIFF_SMOOTH,
  BASELINE_FRAMES, etc.). If the user provides a video file and wants wave heights, 
  this skill applies.
---

# Wave Height Analysis

Orchestrates `calibrate.py`, `analyze.py`, `compute_metrics.py`, and
`generate_report.py` into a single automated workflow.
Computes RMSE/NRMSE benchmark metrics and produces a summary
markdown report.

## Workflow

### 1. Calibration (conditional)

If `calibration.json` does **not** exist in the project root:

```bash
python calibrate.py <video_path>
```

- Opens an interactive OpenCV GUI (**requires a display**)
- User draws ROI tightly around the tape column, clicks two reference points
- Produces `calibration.json`

If `calibration.json` already exists, skip this step entirely.

### 2. Analysis (always)

```bash
python analyze.py <video_path> <output_csv>
```

Produces:
- `<output_csv>` — columns: `frame_number`, `timestamp_s`, `wave_height_cm`
- `wave_height_plot.png` — visualization

### 3. Benchmark metrics (if benchmark available)

If a benchmark CSV exists (same timestamp alignment), compute accuracy:

```bash
python .opencode/skills/wave-height-analysis/scripts/compute_metrics.py \
  <output_csv> <benchmark_csv> [metrics.json]
```

Prints RMSE (cm) and NRMSE (%) to stdout, optionally saves `metrics.json`.

### 4. Generate Report (always)

Summarize detection results and optional benchmark metrics into a markdown file:

```bash
python .opencode/skills/wave-height-analysis/scripts/generate_report.py \
  <output_csv> [report.md]
```

Default report path is `wave_height_report.md`. Overwrites any existing file.
If `metrics.json` exists (produced by step 3), RMSE and NRMSE are included.

Produces:
- `<report>` — markdown report with min/max/mean/median/std heights, frame count,
  duration, calibration info, detection parameters, and optional RMSE/NRMSE

## Tunable Parameters

These constants in `analyze.py` can be modified to improve accuracy.
Refer to `AGENTS.md` for full domain context.

| Constant | Default | Effect |
|----------|---------|--------|
| `TAPE_EDGE_COLS` | 60 | Leftmost ROI columns for median aggregation |
| `BASELINE_FRAMES` | 100 | Frames averaged for still-water brightness profile |
| `DIFF_SMOOTH` | 15 | Smoothing kernel (px) — higher = less turbulence noise |
| `MIN_DIFF_CONFIDENCE` | 4 | Minimum peak abs diff to confirm detection; else treat as baseline |
| `SEARCH_START_FRAC` | 0.50 | Start search as fraction of ROI height (avoids mist) |
| `SEARCH_END_FRAC` | 0.95 | End search as fraction of ROI height |

## Self-Improvement (Autoresearch Loop)

This skill supports autonomous parameter tuning using the karpathy/autoresearch
convention. An LLM agent iteratively modifies `analyze.py`, runs the pipeline,
evaluates against a benchmark, and keeps/discards changes.

### One-time setup

```bash
python .opencode/skills/wave-height-analysis/scripts/prep_benchmark.py PhaseII_TestD_0003_h_01.csv
```

### Experiment loop

Open this repo in an LLM agent (Claude Code, Codex CLI, etc.) and point it to:

```
.opencode/skills/wave-height-analysis/program.md
```

The agent will follow the instructions in `program.md` to autonomously tune
the detection parameters and improve accuracy.

### Files involved

| File | Role |
|------|------|
| `analyze.py` | The file the agent edits (tunable parameters + algorithm) |
| `.opencode/skills/wave-height-analysis/program.md` | Agent instructions for the experiment loop |
| `.opencode/skills/wave-height-analysis/scripts/prep_benchmark.py` | One-time benchmark prep |
| `.opencode/skills/wave-height-analysis/scripts/compute_metrics.py` | Per-experiment metric computation |
| `results.tsv` | Experiment log (not committed) |

## Gotchas

- **Calibration requires a display.** In headless/remote environments, ensure `calibration.json` exists before running.
- **ROI must be tight on the tape column.** Existing `calibration.json` has `w=1131` (too wide). Re-calibrate with a ~60px wide ROI.
- **Water is dark brown/gray, not blue.** No usable hue signal — detection relies on intensity (darkness) of the water mass. The algorithm uses absolute-difference + gradient, not intensity thresholding.
- **Right ~20-25% of frame** is bright blue foam board — a false positive hazard. ROI must exclude this area.
- **Surface is jagged/turbulent** with foam, spray, and mist. Column-wise median aggregation is mandatory (`TAPE_EDGE_COLS`).
- The skill detects the **tape edge** region — not the entire frame. Keep ROI narrow.
