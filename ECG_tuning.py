#!/usr/bin/env python3
"""
Hyper-parameter sweep for harmonic exponent n on ECG200.

• Exponents tested : 1,2,4,8,16,32
• Repeats per n    : 3 seeds  (42,123,999)
• Training script  : time_series.py   (must print a line
  'RESULTS: {...}' with keys harmax_tr, harmax_te, soft_tr, soft_te)

Outputs
========
1) results_harmax_tuning.jsonl  – one JSON dict per (n,seed) run
2) results_harmax_tuning.csv    – same as above, CSV
3) results_harmax_summary.csv   – mean ± std per n  + ratio columns
"""

import subprocess, json, csv, math, pathlib, statistics, sys

# ------------------------------------------------------------------ #
# CONFIG                                                             #
# ------------------------------------------------------------------ #
import os
os.makedirs('data',exist_ok=True)
TRAIN_FILE   = pathlib.Path(__file__).parent / "time_series.py"
OUT_DETAIL_CSV  = pathlib.Path("data/results_harmax_tuning.csv")
OUT_DETAIL_JSON = pathlib.Path("data/results_harmax_tuning.jsonl")
OUT_SUMMARY_CSV = pathlib.Path("data/results_harmax_summary.csv")

EPOCHS          = 400                   # keep in sync with training script
N_EXP_VALUES    = [1, 2, 4, 8, 16]
SEEDS           = [42, 123, 999]        # 3 repetitions
DEVICE          = "cpu"                 # reproducible & fast enough

# ------------------------------------------------------------------ #
# HELPER                                                             #
# ------------------------------------------------------------------ #
def run_training(n_exp: int, seed: int) -> dict:
    """Launch one training run; return dict with mean acc over last half."""
    cmd = [
        sys.executable, str(TRAIN_FILE),
        "--epochs",  str(EPOCHS),
        "--n_exp",   str(n_exp),
        "--seed",    str(seed),
        "--device",  DEVICE,
    ]
    out = subprocess.check_output(cmd, text=True)

    try:
        res_line = next(l for l in out.splitlines() if "RESULTS:" in l)
    except StopIteration:
        raise RuntimeError("RESULTS line not found in script output")

    blobs = json.loads(res_line.split("RESULTS:")[1].strip())

    start = EPOCHS // 2
    def avg(lst): return sum(lst[start:]) / (EPOCHS - start)
    def sd (lst): return statistics.stdev(lst[start:])

    return dict(
        n_exp      = n_exp,
        seed       = seed,
        mean_train_h = round(avg(blobs["harmax_tr"]), 4),
        std_train_h  = round(sd (blobs["harmax_tr"]), 4),
        mean_test_h  = round(avg(blobs["harmax_te"]), 4),
        std_test_h   = round(sd (blobs["harmax_te"]), 4),
        mean_train_s = round(avg(blobs["soft_tr"]),   4),
        std_train_s  = round(sd (blobs["soft_tr"]),   4),
        mean_test_s  = round(avg(blobs["soft_te"]),   4),
        std_test_s   = round(sd (blobs["soft_te"]),   4),
    )

# ------------------------------------------------------------------ #
# MAIN LOOP                                                          #
# ------------------------------------------------------------------ #
detail_rows = []

FIELDNAMES = [
    "n_exp","seed",
    "mean_train_h","std_train_h","mean_test_h","std_test_h",
    "mean_train_s","std_train_s","mean_test_s","std_test_s"
]

OUT_DETAIL_CSV.parent.mkdir(exist_ok=True)

with OUT_DETAIL_CSV.open("w", newline="") as f_csv, \
     OUT_DETAIL_JSON.open("w") as f_json:
    writer = csv.DictWriter(f_csv, fieldnames=FIELDNAMES)
    writer.writeheader()

    for n in N_EXP_VALUES:
        for seed in SEEDS:
            print(f"→ running  n={n:>2}  seed={seed}")
            row = run_training(n, seed)
            writer.writerow(row)
            f_json.write(json.dumps(row) + "\n")
            detail_rows.append(row)

print(f"\nSaved per-run logs to:\n  {OUT_DETAIL_CSV}\n  {OUT_DETAIL_JSON}")

# ------------------------------------------------------------------ #
# AGGREGATE and compute ratios                                       #
# ------------------------------------------------------------------ #
import pandas as pd
df = pd.DataFrame(detail_rows)

agg = df.groupby("n_exp").agg({
    "mean_train_h": ["mean","std"],
    "mean_test_h" : ["mean","std"],
    "mean_train_s": ["mean","std"],
    "mean_test_s" : ["mean","std"],
})

agg.columns = ["_".join(col) for col in agg.columns]   # flatten MultiIndex

# ratios
agg["ratio_mean_train"] = agg["mean_train_h_mean"] / agg["mean_train_s_mean"]
agg["ratio_std_train"]  = agg["mean_train_h_std"]  / agg["mean_train_s_std"]
agg["ratio_mean_test"]  = agg["mean_test_h_mean"]  / agg["mean_test_s_mean"]
agg["ratio_std_test"]   = agg["mean_test_h_std"]   / agg["mean_test_s_std"]

agg.round(4).to_csv(OUT_SUMMARY_CSV)
print("\n=== summary per n_exp ===")
print(agg.round(3).to_markdown())

print(f"\nSaved summary table to  {OUT_SUMMARY_CSV}")
