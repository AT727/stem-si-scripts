import csv
import sys
import numpy as np

def load_first(path):
    data = []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.reader(f):
            if len(r) >= 4 and all(c.strip() for c in r[:4]):
                try:
                    t = float(r[0])
                    cy = float(r[3])
                    data.append((t, cy * 100))
                except ValueError:
                    pass
    return data

def load_benchmark(path):
    data = []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.reader(f):
            if len(r) >= 4 and all(c.strip() for c in r[:4]):
                try:
                    t = float(r[1])
                    yc = float(r[3])
                    data.append((t, yc))
                except ValueError:
                    pass
    return data

def align(preds, targets, tol=0.01):
    p_vals, t_vals = [], []
    j = 0
    for t, p in preds:
        while j < len(targets) and targets[j][0] < t - tol:
            j += 1
        if j < len(targets) and abs(targets[j][0] - t) < tol:
            t_vals.append(targets[j][1])
            p_vals.append(p)
    return np.array(p_vals), np.array(t_vals)

if __name__ == "__main__":
    first_path = sys.argv[1] if len(sys.argv) > 1 else "first.csv"
    bench_path = sys.argv[2] if len(sys.argv) > 2 else "benchmark.csv"

    preds = load_first(first_path)
    targets = load_benchmark(bench_path)
    p, t = align(preds, targets)

    rmse = np.sqrt(np.mean((p - t) ** 2))
    nrmse = rmse / (t.max() - t.min()) * 100

    print(f"Matched: {len(p)} points")
    print(f"Pred range: [{p.min():.3f}, {p.max():.3f}] cm")
    print(f"Target range: [{t.min():.3f}, {t.max():.3f}] cm")
    print(f"RMSE:  {rmse:.6f} cm")
    print(f"NRMSE: {nrmse:.4f}%")
