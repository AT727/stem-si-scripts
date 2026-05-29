import sys
import pandas as pd
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python prep_benchmark.py <raw_benchmark.csv> [output.csv]")
    sys.exit(1)

raw_path = Path(sys.argv[1])
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("benchmark_aligned.csv")

raw = pd.read_csv(raw_path, skiprows=2, header=None)
raw.columns = ['frame', 't', 'y', 'y_correct']

frame = raw['frame'].astype(int).values
y_correct = raw['y_correct'].astype(float).values

baseline_offset = float(np.mean(y_correct[:27]))
print(f"Baseline y_correct offset (first 27 frames): {baseline_offset:.4f} cm")

wave_height = y_correct - baseline_offset

df = pd.DataFrame({'frame_number': frame, 'wave_height_cm': np.round(wave_height, 4)})
df.to_csv(output_path, index=False)
print(f"Benchmark aligned: {output_path} ({len(df)} frames)")
print(f"  Height range: {wave_height.min():.3f} to {wave_height.max():.3f} cm")
