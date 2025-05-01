#!/usr/bin/env python3
import subprocess
import json
from pathlib import Path

import pandas as pd

# ── 1) run temp.py over your list of datasets ──────────────────────

datasets = [
    "SyntheticControl",
    "ShapeletSim",
    "TwoPatterns",
    "Coffee",
    "OliveOil",
    "GunPoint",
    "ECG5000",
    "ItalyPowerDemand",
    "Wafer",
    "FordA",
]

SCRIPT = "temp.py"
COMMON_ARGS = [
    "--featurize",
    "--epochs",   "200",
    "--batch",    "32",
    "--lr",       "3e-3",
    "--wd",       "0",
]

for ds in datasets:
    print(f"===== Running featurized on {ds} =====")
    cmd = ["python", SCRIPT, "--dataset", ds] + COMMON_ARGS
    subprocess.run(cmd, check=True)


# ── 2) read back all featurized results and build a table ─────────

results_dir = Path("results") / "featureize"
rows = []

for json_path in sorted(results_dir.glob("*_featTrue.json")):
    data = json.loads(json_path.read_text())
    # final test accuracies
    s_test = data["soft_last"]
    h_test = data["har_last"]
    # final train accuracies (last epoch in history)
    hist = data["hist"]
    s_train = hist["softmax"]["tr"][-1]
    h_train = hist["harmax"]["tr"][-1]

    rows.append({
        "dataset":         data["dataset"],
        "softmax_test":    s_test,
        "harmax_test":     h_test,
        "Δ_test":          h_test - s_test,
        "softmax_train":   s_train,
        "harmax_train":    h_train,
        "Δ_train":         h_train - s_train,
    })

df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

# always print a markdown summary
print(df)

