import pandas as pd

bench = pd.read_csv("benchmark_aligned.csv")
print(f"Total benchmark frames: {len(bench)}")
print(f"Frame range: {bench.frame_number.min()}-{bench.frame_number.max()}")

significant = bench[bench["wave_height_cm"] > 0.5]
if len(significant) > 0:
    print(f"First frame > 0.5 cm: {significant.frame_number.iloc[0]}")
else:
    print("No frames > 0.5 cm")

max_idx = bench["wave_height_cm"].idxmax()
print(f"Max height: {bench.loc[max_idx, 'wave_height_cm']:.3f} cm at frame {bench.loc[max_idx, 'frame_number']}")

peak = bench[bench["wave_height_cm"] > 8]
if len(peak) > 0:
    print(f"Peak frames (>8cm): {peak.frame_number.iloc[0]}-{peak.frame_number.iloc[-1]}")

# Find wave start
rising = bench[bench["wave_height_cm"] > 0.1]
if len(rising) > 0:
    print(f"First frame > 0.1 cm: {rising.frame_number.iloc[0]}")

# Print a few benchmark heights around key frames
for f in [100, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 2500, 2800]:
    row = bench[bench["frame_number"] == f]
    if len(row) > 0:
        print(f"  Frame {f}: {row['wave_height_cm'].values[0]:.3f} cm")
