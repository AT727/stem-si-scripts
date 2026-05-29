import pandas as pd

raw = pd.read_csv("PhaseII_TestD_0003_h_01.csv", skiprows=2, header=None)
raw.columns = ['frame', 't', 'y', 'y_correct']

# When does y_correct first exceed baseline+0.1?
baseline = raw['y_correct'].iloc[:27].mean()
print(f"Baseline y_correct: {baseline:.4f}")
raw['wave_h'] = raw['y_correct'] - baseline

rising = raw[raw['wave_h'] > 0.1]
if len(rising) > 0:
    print(f"First frame > 0.1 cm: {rising['frame'].iloc[0]:.0f}")

rising1 = raw[raw['wave_h'] > 1.0]
if len(rising1) > 0:
    print(f"First frame > 1.0 cm: {rising1['frame'].iloc[0]:.0f}")

# Print wave heights at key frames
for f in [150, 180, 190, 200, 250, 300, 350, 400, 500]:
    row = raw[raw['frame'] == f]
    if len(row) > 0:
        print(f"  Frame {f}: raw y_correct={row['y_correct'].values[0]:.3f}, wave_h={row['wave_h'].values[0]:.3f}")

# Also check our detection
pred = pd.read_csv("output.csv")
print(f"\nOur detection at same frames:")
for f in [150, 180, 190, 200, 250, 300, 350, 400, 500]:
    row = pred[pred['frame_number'] == f]
    if len(row) > 0:
        print(f"  Frame {f}: pred={row['wave_height_cm'].values[0]:.3f}")
