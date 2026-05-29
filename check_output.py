import pandas as pd
pred = pd.read_csv("output.csv")
for fn in [150, 180, 190, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000, 1500, 2000, 2500]:
    row = pred[pred['frame_number'] == fn]
    if len(row) > 0:
        print(f"Frame {fn}: {row['wave_height_cm'].values[0]:.4f}")
