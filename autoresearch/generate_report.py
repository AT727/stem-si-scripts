import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


if len(sys.argv) < 2:
    print("Usage: python generate_report.py <output_csv> [report.md]")
    sys.exit(1)


csv_path = Path(sys.argv[1])
report_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("wave_height_report.md")


if not csv_path.exists():
    print(f"Error: CSV not found: {csv_path}")
    sys.exit(1)


df = pd.read_csv(csv_path)
total_frames = len(df)


baseline_frames = (df['wave_height_cm'] == 0).sum() if 'wave_height_cm' in df.columns else 0
analyzed_frames = total_frames - baseline_frames


valid = df['wave_height_cm'].dropna() if 'wave_height_cm' in df.columns else pd.Series([], dtype=float)
detection_failures = df['wave_height_cm'].isna().sum() if 'wave_height_cm' in df.columns else 0


duration_s = df['timestamp_s'].iloc[-1] - df['timestamp_s'].iloc[0] if total_frames > 0 and 'timestamp_s' in df.columns else 0.0


cal = {}
cal_path = Path("calibration.json")
if cal_path.exists():
    with open(cal_path) as f:
        cal = json.load(f)


metrics = {}
metrics_path = csv_path.parent / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)


video_name = csv_path.stem


lines = []
lines.append(f"# Wave Height Analysis Report")
lines.append("")
lines.append(f"**Video:** {video_name}.MP4")
lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("")


lines.append("## Summary")
lines.append("| Metric | Value |")
lines.append("|--------|-------|")
lines.append(f"| Total frames | {total_frames} |")
lines.append(f"| Duration | {duration_s:.2f} s |")
lines.append(f"| Baseline frames | {baseline_frames} |")
lines.append(f"| Analyzed frames | {analyzed_frames} |")
lines.append(f"| Detection failures | {detection_failures} |")
lines.append("")


if len(valid) > 0:
    lines.append("## Wave Heights")
    lines.append("| Statistic | Value (cm) |")
    lines.append("|-----------|------------|")
    lines.append(f"| Min | {valid.min():.3f} |")
    lines.append(f"| Max | {valid.max():.3f} |")
    lines.append(f"| Mean | {valid.mean():.3f} |")
    lines.append(f"| Median | {valid.median():.3f} |")
    lines.append(f"| Std | {valid.std():.3f} |")
    lines.append("")
else:
    lines.append("## Wave Heights")
    lines.append("No valid detections.\n")


if metrics:
    lines.append("## Benchmark Metrics")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| RMSE | {metrics.get('rmse_cm', 'N/A')} cm |")
    lines.append(f"| NRMSE | {metrics.get('nrmse_pct', 'N/A')}% |")
    lines.append("")


if cal:
    lines.append("## Calibration")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| pixels_per_cm | {cal.get('pixels_per_cm', 'N/A')} |")
    roi = cal.get('roi', {})
    if roi:
        lines.append(f"| ROI | ({roi.get('x', '?')}, {roi.get('y', '?')}, {roi.get('w', '?')}, {roi.get('h', '?')}) |")
    lines.append(f"| FPS | {cal.get('fps', 'N/A')} |")
    baseline_y = cal.get('baseline_y')
    if baseline_y is not None:
        lines.append(f"| Baseline (ROI-relative) | {baseline_y} |")
    lines.append("")


lines.append("## Detection Parameters")
lines.append("| Parameter | Default | Effect |")
lines.append("|-----------|---------|--------|")
lines.append("| `TAPE_EDGE_COLS` | 60 | Leftmost ROI columns for median aggregation |")
lines.append("| `BASELINE_FRAMES` | 100 | Frames averaged for still-water brightness profile |")
lines.append("| `DIFF_SMOOTH` | 15 | Smoothing kernel (px) |")
lines.append("| `MIN_DIFF_CONFIDENCE` | 4 | Minimum peak abs diff to confirm detection |")
lines.append("| `SEARCH_START_FRAC` | 0.50 | Start search as fraction of ROI height |")
lines.append("| `SEARCH_END_FRAC` | 0.95 | End search as fraction of ROI height |")


report_text = "\n".join(lines) + "\n"
report_path.write_text(report_text, encoding='utf-8')
print(f"Report saved to {report_path}")



