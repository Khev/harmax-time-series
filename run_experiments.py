#!/usr/bin/env python3
"""
Run a mini-suite of UCR datasets, cache metrics, and emit CSV + LaTeX tables.
"""

import argparse, subprocess, json, csv, pathlib, time, sys, logging
from datetime import datetime

TRAIN_SCRIPT = "ucr_harmax_versus_softmax.py"

# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def cache_path(ds: str, seed: int, n_exp: int) -> pathlib.Path:
    return pathlib.Path("results") / f"{ds}_seed{seed}_n_harmonic_{n_exp}.json"

def run_dataset(ds: str, seed: int, epochs: int, n_exp: int):
    """Execute training unless cached metrics already exist."""
    if cache_path(ds, seed, n_exp).exists():
        logging.info("[%-15s] cached – skipping", ds)
        return

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--dataset", ds,
        "--seed",    str(seed),
        "--epochs",  str(epochs),
        "--n_exp",   str(n_exp)
    ]
    logging.info("▶ %s", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    logging.info("[%-15s] finished in %.1fs", ds, time.time() - t0)

def load_metrics(ds: str, seed: int, n_exp: int):
    with cache_path(ds, seed, n_exp).open() as f:
        return json.load(f)

# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[
        "ECG200", "Coffee", "GunPoint", "ItalyPowerDemand",
        "ShapeletSim", "Plane", "OliveOil", "FordA",
        "StarLightCurves", "HandOutlines",
    ])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--n_exp",  type=int, default=7,
                    help="harmonic exponent n passed to the training script")
    args = ap.parse_args()

    pathlib.Path("results").mkdir(exist_ok=True)

    # 1 train / reuse
    for ds in args.datasets:
        run_dataset(ds, args.seed, args.epochs, args.n_exp)

    # 2 aggregate
    rows = []
    for ds in args.datasets:
        m = load_metrics(ds, args.seed, args.n_exp)
        rows.append({
            "dataset": ds,
            "soft":    m["soft_last_te"],
            "har":     m["har_last_te"],
            "delta":   round(m["har_last_te"] - m["soft_last_te"], 4),
        })

    # 3 CSV
    csv_path = pathlib.Path("results") / "ucr_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "soft", "har", "delta"])
        w.writeheader(); w.writerows(rows)
    logging.info("Summary CSV written to %s", csv_path.resolve())

    # 4 Markdown table (console)
    print("\n```markdown")
    print("| dataset | softmax | harmax | Δ |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['soft']:.3f} | {r['har']:.3f} | {r['delta']:+.3f} |")
    print("```")

    # 5 LaTeX table
    tex_lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Soft-max vs. Har-max on selected UCR datasets}",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Dataset & Softmax & Harmax & $\\Delta$ \\\\",
        "    \\midrule",
    ]
    tex_lines += [
        f"    {r['dataset']} & {r['soft']:.3f} & {r['har']:.3f} & {r['delta']:+.3f} \\\\"
        for r in rows
    ]
    tex_lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\label{tab:ucr_harmax}",
        "\\end{table}",
        "",
    ]
    tex_path = pathlib.Path("results") / "ucr_summary.tex"
    tex_path.write_text("\n".join(tex_lines))
    logging.info("LaTeX table written to %s", tex_path.resolve())

if __name__ == "__main__":
    main()
