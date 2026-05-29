import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

TSV = Path(__file__).parent / "results.tsv"
OUT = Path(__file__).parent / "progress.png"

df = pd.read_table(TSV)
if len(df) < 2:
    print("Need at least 2 experiments for a progress chart")
    exit(1)

df = df.reset_index(drop=True)
x = np.arange(len(df))

fig, ax1 = plt.subplots(figsize=(10, 5))

colors = df["status"].map({"keep": "#2ca02c", "discard": "#d62728"}).fillna("#7f7f7f")
markers = df["status"].map(lambda s: "o" if s == "keep" else "x")

for i in x:
    ax1.scatter(i, df.loc[i, "rmse"], c=colors.iloc[i], marker=markers.iloc[i], s=60, zorder=5)

ax1.plot(x, df["rmse"], color="#1f77b4", alpha=0.4, linewidth=1, zorder=1)
ax1.set_xlabel("Experiment #")
ax1.set_ylabel("RMSE (cm)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

ax2 = ax1.twinx()
ax2.plot(x, df["nrmse_pct"], color="#ff7f0e", alpha=0.6, linewidth=1, linestyle="--", label="NRMSE %")
ax2.scatter(x, df["nrmse_pct"], color="#ff7f0e", marker="s", s=40, zorder=4)
ax2.axhline(5, color="#ff7f0e", linewidth=0.8, linestyle=":", alpha=0.7)
ax2.set_ylabel("NRMSE (%)", color="#ff7f0e")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")

last_best_rmse = None
best_idx = None
for i in x:
    status = df.loc[i, "status"]
    if status == "keep":
        passes = df.loc[i, "description"] if pd.notna(df.loc[i, "description"]) else ""
        is_pass = "pass" in str(passes).lower()
        rmse = df.loc[i, "rmse"]
        if last_best_rmse is None or rmse < last_best_rmse:
            last_best_rmse = rmse
            best_idx = i

if best_idx is not None:
    ax1.scatter(best_idx, df.loc[best_idx, "rmse"], facecolors="none",
                edgecolors="gold", linewidths=2.5, s=140, zorder=6, label="Best RMSE")

handles = [
    plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2ca02c", markersize=8, label="Keep"),
    plt.Line2D([], [], marker="x", color="w", markerfacecolor="#d62728", markersize=8, label="Discard"),
]
ax1.legend(handles=handles, loc="upper left")

fig.suptitle("Wave Height Detection — Autoresearch Progress", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}")
