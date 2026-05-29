# Benchmarking

Ground-truth CSV used to score detection accuracy.

## CSV Format

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_s` | float | Timestamp in seconds, matching analyze.py output |
| `wave_height_cm` | float | Ground truth wave height in cm |

### Example

```csv
timestamp_s,wave_height_cm
0.0,0.0
0.0083,0.5
0.0167,1.2
0.0250,-0.3
```

## Metrics

### RMSE (cm)

Root Mean Square Error between predicted and benchmark wave heights.

```
RMSE = sqrt(mean((predicted - benchmark)^2))
```

Lower is better. Represents typical error magnitude in centimeters. A RMSE of 0.5 cm means predictions are off by ~0.5 cm on average.

### NRMSE (%)

Normalized RMSE, expressed as a percentage of the benchmark data range.

```
NRMSE = (RMSE / (max(benchmark) - min(benchmark))) * 100
```

Useful for comparing accuracy across different wave conditions (e.g., calm vs. stormy videos). Lower is better. A NRMSE of 5% means the typical error is 5% of the full wave height range.
