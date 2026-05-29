import pandas as pd
import numpy as np

det = pd.read_csv('output.csv')
ben = pd.read_csv('benchmark_aligned.csv')
m = pd.merge(det, ben, on='frame_number', suffixes=('_d', '_b'))
m['e'] = abs(m['wave_height_cm_d'] - m['wave_height_cm_b'])
print('Top 20 largest errors:')
print(m.sort_values('e', ascending=False).head(20)[['frame_number', 'wave_height_cm_d', 'wave_height_cm_b', 'e']].to_string())
print()
print(f'Total frames: {len(m)}')
print(f'Frames with e > 5: {(m["e"] > 5).sum()}')
print(f'Frames with e > 3: {(m["e"] > 3).sum()}')
print(f'Frames with e > 1: {(m["e"] > 1).sum()}')
print(f'Frames with e < 0.5: {(m["e"] < 0.5).sum()}')
sq_err = (m['wave_height_cm_d'] - m['wave_height_cm_b']) ** 2
print(f'RMSE: {np.sqrt(sq_err.mean()):.4f}')

# Contribution to MSE from different error regimes
total_mse = sq_err.mean()
crest = m[m['e'] > 5]
trough = m[m['e'] > 3]
print(f'\nMSE contribution of crest (e>5, n={len(crest)}): {sq_err[crest.index].sum() / len(m):.4f}')
print(f'MSE contribution of e>3 (n={len(trough)}): {sq_err[trough.index].sum() / len(m):.4f}')
print(f'Rest of frames (n={len(m)-len(trough)}): {(sq_err.sum() - sq_err[trough.index].sum()) / len(m):.4f}')
