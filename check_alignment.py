import pandas as pd

# Check raw Tracker CSV
raw = pd.read_csv("PhaseII_TestD_0003_h_01.csv", skiprows=2, header=None)
raw.columns = ['frame', 't', 'y', 'y_correct']
print("Raw Tracker CSV:")
print(f"  Frames: {raw['frame'].min()} - {raw['frame'].max()} ({len(raw)} total)")
print(f"  First 5:")
print(raw.head(5).to_string(index=False))
print(f"  Frame 100-105:")
print(raw.iloc[100:106].to_string(index=False))
print(f"  Frame 186-195:")
print(raw.iloc[186:196].to_string(index=False))

# Check aligned benchmark
bench = pd.read_csv("benchmark_aligned.csv")
print(f"\nAligned benchmark:")
print(f"  Frames: {bench['frame_number'].min()} - {bench['frame_number'].max()} ({len(bench)} total)")
print(f"  Frame 186-195:")
rows = bench[bench['frame_number'].between(186, 195)]
print(rows.to_string(index=False))
