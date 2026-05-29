# Wave Height Analysis Report

**Video:** output.MP4
**Generated:** 2026-05-29 05:43:17

## Summary
| Metric | Value |
|--------|-------|
| Total frames | 2981 |
| Duration | 24.86 s |
| Baseline frames | 100 |
| Analyzed frames | 2881 |
| Detection failures | 0 |

## Wave Heights
| Statistic | Value (cm) |
|-----------|------------|
| Min | 0.000 |
| Max | 5.638 |
| Mean | 3.962 |
| Median | 3.944 |
| Std | 1.217 |

## Benchmark Metrics
| Metric | Value |
|--------|-------|
| RMSE | 2.7397 cm |
| NRMSE | 33.4112% |

## Calibration
| Parameter | Value |
|-----------|-------|
| pixels_per_cm | 77.33 |
| ROI | (1758, 1191, 105, 969) |
| FPS | 119.88011988011988 |
| Baseline (ROI-relative) | 937 |

## Detection Parameters
| Parameter | Default | Effect |
|-----------|---------|--------|
| `TAPE_EDGE_COLS` | 60 | Leftmost ROI columns for median aggregation |
| `BASELINE_FRAMES` | 100 | Frames averaged for still-water brightness profile |
| `DIFF_SMOOTH` | 15 | Smoothing kernel (px) |
| `MIN_DIFF_CONFIDENCE` | 4 | Minimum peak abs diff to confirm detection |
| `SEARCH_START_FRAC` | 0.50 | Start search as fraction of ROI height |
| `SEARCH_END_FRAC` | 0.95 | End search as fraction of ROI height |
