---
name: wave-error-analysis
description: Calculates RMSE (cm) and NRMSE (%) for wave height analysis results. Use when comparing output.csv data against ground truth data.
---

# Wave Error Analysis

## Overview

This skill provides a script to calculate Root Mean Square Error (RMSE) and Normalized Root Mean Square Error (NRMSE) for comparing experimental wave height measurements (`output.csv`) against ground truth measurements.

## Usage

Use this skill when you need to validate the accuracy of wave height measurements derived from analysis pipelines.

### Calculation

Ensure you have both:
1. `output.csv`: The output from `analyze.py`.
2. `PhaseII_TestD_0003_h_01.csv`: Your ground truth dataset.

Both files must have `frame_number` and `wave_height_cm` columns.

Run the calculation:
```bash
python scripts/compute_metrics.py output.csv truth.csv
```

