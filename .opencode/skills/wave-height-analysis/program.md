# Wave Height Auto-Research

An autonomous experiment loop that tunes `analyze.py` to minimize RMSE against
a manual ground-truth benchmark (Tracker CSV). Modeled after karpathy/autoresearch.

## Setup

Before starting, verify:

1. **Inputs exist:**
   - `PhaseII_TestD_0003_c4_01.MP4` — test video
   - `benchmark_aligned.csv` — ground truth (run `prep_benchmark.py` if missing)
   - `calibration.json` — ROI and pixel calibration (run `calibrate.py` if missing)

2. **Initialize results.tsv** — create with header if not exists:
   ```
   commit	rmse	nrmse_pct	max_height	min_height	status	description
   ```

3. **Establish baseline:** Run the pipeline once with current `analyze.py` as-is.
   Log the result as `baseline`.

4. **Run `prep_benchmark.py` first** if `benchmark_aligned.csv` doesn't exist:
   ```bash
   python .opencode/skills/wave-height-analysis/scripts/prep_benchmark.py PhaseII_TestD_0003_h_01.csv
   ```

## What you CAN do

- **Edit `analyze.py`** — tune any constant: `TAPE_EDGE_COLS`, `BASELINE_FRAMES`,
  `DIFF_SMOOTH`, `MIN_DIFF_CONFIDENCE`, `SEARCH_START_FRAC`, `SEARCH_END_FRAC`.
  Also change the detection algorithm itself if you have a better idea.
- **Edit any file under `.opencode/skills/wave-height-analysis/`** — scripts,
  references, etc.

## What you CANNOT do

- Modify `calibrate.py` — it is fixed. Only re-run if calibration.json is missing.
- Install new packages — only opencv-python, numpy, pandas, matplotlib.
- Change the benchmark or metrics computation in a way that invalidates comparisons.
- Ask the human for permission mid-loop — be autonomous.

## The objective metric

A **pass** requires ALL five constraints satisfied simultaneously:

1. **RMSE** — as low as possible (primary optimization target)
2. **NRMSE** — within 5% (hard constraint)
3. **Max height** — between 8 and 9 cm (matches benchmark peak ~8.3 cm)
4. **Min height** — ≤ 0.1 cm (baseline should be near zero)
5. **No detection failures** — all frames must produce a valid height

Among passes, lower RMSE wins. If no pass exists, prioritize satisfying the
constraints over minimizing RMSE.

## Running one experiment

```bash
python analyze.py PhaseII_TestD_0003_c4_01.MP4 output.csv
python .opencode/skills/wave-height-analysis/scripts/compute_metrics.py output.csv benchmark_aligned.csv metrics.json
```

Read the metrics: `cat metrics.json`

## The experiment loop

LOOP FOREVER:

1. **Read state** — `git log --oneline -3`, `tail -5 results.tsv`, `cat metrics.json`
2. **Brainstorm** — What parameter change or algorithmic tweak might improve RMSE
   while keeping NRMSE < 5% and heights in range?
3. **Edit** — Modify `analyze.py` with your experiment
4. **Commit** — `git commit -am "experiment: <short description>"`
5. **Run** — `python analyze.py PhaseII_TestD_0003_c4_01.MP4 output.csv > run.log 2>&1`
6. **Evaluate** — `python .opencode/skills/wave-height-analysis/scripts/compute_metrics.py output.csv benchmark_aligned.csv metrics.json`
7. **Read results** — Extract rmse, nrmse_pct, max_height, min_height from stdout or metrics.json
8. **Log** — Append to `results.tsv` (tab-separated, do NOT commit this file):
   ```
   <7-char-commit>	<rmse>	<nrmse_pct>	<max_height>	<min_height>	keep|discard	<description>
   ```
9. **Decide:**
   - If ALL constraints satisfied AND RMSE improved vs best known → `git branch -f best HEAD` (save checkpoint), status = `keep`
   - If constraints violated OR RMSE worse than best → `git reset --hard HEAD~1`, status = `discard`
   - If crash → fix and re-run, or status = `crash` and move on
10. **Repeat** — Do NOT ask "should I continue?" The answer is always yes.

## Simplicity criterion

All else being equal, simpler is better. A 0.01 cm RMSE improvement that adds
20 lines of fragile code is not worth it. A 0.01 cm RMSE improvement from
deleting code or simplifying a threshold is a clear win.

## Troubleshooting

- **Crash:** check `tail -50 run.log` for the traceback. Fix typos/packaging
  issues and re-run. If the idea itself is broken, discard and move on.
- **Bad heights (e.g., 10+ cm):** The water surface is being detected too high
  (confusing spray/mist for water). Try adjusting `SEARCH_START_FRAC` or
  increasing `DIFF_SMOOTH`.
- **All zeros:** The gradient isn't finding the surface. Check `MIN_DIFF_CONFIDENCE`
  and `DIFF_SMOOTH`.
- **Negative heights that should be positive:** The baseline row is wrong. Check
  `SEARCH_END_FRAC`.
