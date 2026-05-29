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
metrics = {'rmse_cm': round(rmse, 4), 'nrmse_pct': round(nrmse, 4)}
print(f"Merged rows: {len(merged)}")
print(f"Valid rows: {len(valid)}")
print(f"RMSE:  {rmse:.4f} cm")
print(f"NRMSE: {nrmse:.2f}%")
if len(sys.argv) > 3:
    with open(sys.argv[3], 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {sys.argv[3]}")
