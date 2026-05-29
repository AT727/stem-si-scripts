import sys
import json
import pandas as pd
import numpy as np

pred = pd.read_csv(sys.argv[1])
bench = pd.read_csv(sys.argv[2])

merged = pd.merge(pred, bench, on='frame_number', suffixes=('_pred', '_bench'))
valid = merged.dropna(subset=['wave_height_cm_pred', 'wave_height_cm_bench'])

rmse = float(np.sqrt(np.mean((valid['wave_height_cm_pred'] - valid['wave_height_cm_bench']) ** 2)))
bench_range = valid['wave_height_cm_bench'].max() - valid['wave_height_cm_bench'].min()
nrmse = float((rmse / bench_range * 100)) if bench_range != 0 else 0.0
max_h = float(valid['wave_height_cm_pred'].max())
min_h = float(valid['wave_height_cm_pred'].min())

bench_max = float(valid['wave_height_cm_bench'].max())
bench_min = float(valid['wave_height_cm_bench'].min())

metrics = {
    'rmse_cm': round(rmse, 4),
    'nrmse_pct': round(nrmse, 4),
    'max_height_cm': round(max_h, 4),
    'min_height_cm': round(min_h, 4),
    'bench_max_height_cm': round(bench_max, 4),
    'bench_min_height_cm': round(bench_min, 4),
    'detection_failures': int(merged['wave_height_cm_pred'].isna().sum())
}

print(f"RMSE:  {rmse:.4f} cm")
print(f"NRMSE: {nrmse:.2f}%")
print(f"Max height: {max_h:.3f} cm  (benchmark: {bench_max:.3f})")
print(f"Min height: {min_h:.3f} cm  (benchmark: {bench_min:.3f})")
print(f"Detection failures: {metrics['detection_failures']}")

all_pass = True
if nrmse > 5.0:
    print("FAIL: NRMSE > 5%")
    all_pass = False
if not (8.0 <= max_h <= 9.0):
    print(f"FAIL: Max height {max_h:.2f} not in [8.0, 9.0]")
    all_pass = False
if min_h > 0.1:
    print(f"FAIL: Min height {min_h:.3f} > 0.1")
    all_pass = False
if metrics['detection_failures'] > 0:
    print(f"FAIL: {metrics['detection_failures']} detection failures")
    all_pass = False

print(f"ALL CONSTRAINTS: {'PASS' if all_pass else 'FAIL'}")

if len(sys.argv) > 3:
    with open(sys.argv[3], 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {sys.argv[3]}")
