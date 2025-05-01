#!/usr/bin/env python3
"""
Run a mini‐suite of UCR datasets, cache metrics under L_p subfolders,
and emit CSV + LaTeX tables in results/L_<lp>/
"""
import argparse
import subprocess
import json
import csv
import pathlib
import time
import sys
import logging

TRAIN_SCRIPT = "temp.py"

# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def cache_path(ds: str, seed: int, n_exp: int, lp: float) -> pathlib.Path:
    outdir = pathlib.Path("results") / f"L_{lp}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{ds}_seed{seed}_n{n_exp}.json"

def run_dataset(ds: str, seed: int, epochs: int, n_exp: int, lp: float):
    """Execute training unless cached metrics already exist."""
    pth = cache_path(ds, seed, n_exp, lp)
    if pth.exists():
        logging.info("[%-15s] cached – skipping", ds)
        return

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--dataset", ds,
        "--seed",    str(seed),
        "--epochs",  str(epochs),
        "--n_exp",   str(n_exp),
        "--lp",      str(lp),
        "--featurize"
    ]
    logging.info("▶ %s", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    mins = int(elapsed // 60)
    secs = elapsed % 60
    logging.info("[%-15s] finished in %dm%.0fs", ds, mins, secs)

def load_metrics(ds: str, seed: int, n_exp: int, lp: float):
    with cache_path(ds, seed, n_exp, lp).open() as f:
        return json.load(f)

# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets", nargs="+", default=[
            "SyntheticControl","ShapeletSim","TwoPatterns",
            "Coffee","OliveOil","GunPoint","ECG5000",
            "ItalyPowerDemand","Wafer","FordA"
        ])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--n_exp",  type=int, default=1,
                    help="harmonic exponent n")
    ap.add_argument("--lp",     type=float, default=2.0,
                    help="Lp norm to use in HarMax (passed to TRAIN_SCRIPT)")
    args = ap.parse_args()

    # 1. train or skip
    for ds in args.datasets:
        run_dataset(ds, args.seed, args.epochs, args.n_exp, args.lp)

    # 2. aggregate
    rows = []
    for ds in args.datasets:
        m = load_metrics(ds, args.seed, args.n_exp, args.lp)
        rows.append({
            "dataset": ds,
            "soft":    m["soft_last_te"],
            "har":     m["har_last_te"],
            "delta":   m["har_last_te"] - m["soft_last_te"],
        })

    # 3. CSV
    outdir = pathlib.Path("results")/f"L_{args.lp}"
    csv_path = outdir/"ucr_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, ["dataset","soft","har","delta"])
        w.writeheader()
        w.writerows(rows)
    logging.info("Summary CSV written to %s", csv_path.resolve())

    # 4. Markdown
    print("\n```markdown")
    print("| dataset | softmax | harmax | Δ |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['dataset']} | {r['soft']:.3f} | {r['har']:.3f} | {r['delta']:+.3f} |")
    print("```")

    # 5. LaTeX
    tex = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Softmax vs Harmax ($L_{" + str(args.lp) + "}$) on UCR}",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Dataset & Softmax & Harmax & $\\Delta$ \\\\",
        "    \\midrule",
    ]
    for r in rows:
        tex.append(
            f"    {r['dataset']} & {r['soft']:.3f} & {r['har']:.3f} & {r['delta']:+.3f} \\\\"
        )
    tex += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\label{tab:ucr_L" + str(args.lp) + "}",
        "\\end{table}",
    ]
    tex_path = outdir/"ucr_summary.tex"
    tex_path.write_text("\n".join(tex))
    logging.info("LaTeX table written to %s", tex_path.resolve())

if __name__ == "__main__":
    main()
