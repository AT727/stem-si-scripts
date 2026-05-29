import pandas as pd
import numpy as np
import sys

def compute_metrics(output_csv, truth_csv):
    # Load CSVs
    # test_output.csv: frame_number, timestamp_s, wave_height_cm, wave_height_filtered_cm
    # truth_csv: frame, t, y, y correct
    df_out = pd.read_csv(output_csv)
    df_truth = pd.read_csv(truth_csv, skiprows=1) # Skip mass_A header

    # Rename to align
    df_out = df_out.rename(columns={'frame_number': 'frame', 'wave_height_cm': 'y_cm'})
    df_truth = df_truth.rename(columns={'frame': 'frame', 'y correct': 'y_cm'})

    # Join on frame to ensure alignment
    df = pd.merge(df_out, df_truth, on='frame', suffixes=('_out', '_truth'))

    # Calc RMSE
    mse = np.mean((df['y_cm_out'] - df['y_cm_truth'])**2)
    rmse = np.sqrt(mse)
    
    # Calc NRMSE (using range: max - min of truth)
    range_val = df['y_cm_truth'].max() - df['y_cm_truth'].min()
    nrmse = (rmse / range_val) * 100

    print(f"RMSE: {rmse:.4f} cm")
    print(f"NRMSE: {nrmse:.2f} %")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compute_metrics.py <output.csv> <truth.csv>")
        sys.exit(1)
    compute_metrics(sys.argv[1], sys.argv[2])
