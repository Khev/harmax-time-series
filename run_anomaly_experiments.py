#!/usr/bin/env python3
"""
Run one‐class “anomaly” detection with HarMax vs Softmax across multiple
UCR (or synthetic) datasets and seeds, save per‐run results & plots, then
aggregate into mean±std and emit a LaTeX table.
"""
import argparse
import json
import logging
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve
)
from torch.utils.data import DataLoader, TensorDataset

# optional aeon loader
try:
    from aeon.datasets import load_classification
except ImportError:
    load_classification = None

# for synthetic toy sets
from utils.utils_synthetic_datasets import make_synthetic_dataset


# ─── data loading utilities ───────────────────────────────────────────

def load_ucr_txt(path: pathlib.Path):
    series, labels = [], []
    with open(path) as f:
        for ln in f:
            parts = ln.strip().replace("\t", " ").split(",")
            if len(parts) == 1:
                parts = ln.split()
            lab, *vals = parts
            labels.append(float(lab))
            series.append([float(v) for v in vals])
    X = np.array(series, dtype=np.float32)
    y_raw = np.array(labels, dtype=np.int64)
    # remap to 0…C-1
    uniq = {v: i for i, v in enumerate(sorted(set(y_raw)))}
    y = np.vectorize(uniq.get)(y_raw).astype(np.int64)
    return X, y

def load_dataset_aeon(name: str, split="train"):
    if load_classification is None:
        raise RuntimeError("aeon not installed; cannot load via aeon")
    try:
        X, y = load_classification(name, split=split, return_type="numpy3d")
    except TypeError:
        X, y = load_classification(name, split=split)
    X = np.asarray(X)
    if X.ndim == 3:
        X = X[:, 0, :]
    return X.astype(np.float32), y.astype(np.int64)


# ─── model & training bits ────────────────────────────────────────────

class Encoder1D(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, d), nn.ReLU()
        )
    def forward(self, x):
        # x: (B, L) → embed (B, d)
        return self.net(x.unsqueeze(1))

class SoftmaxHead(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.logits = nn.Linear(d, k)
    def forward(self, h): return self.logits(h)

class HarmaxHead(nn.Module):
    def __init__(self, d, k, n_exp=None):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(k, d) / d**0.5)
        self.n_exp = int(n_exp) if n_exp else int(d**0.5)
    def forward(self, h):
        # h: (B, d), centres: (k, d)
        dist = torch.cdist(h, self.centres) + 1e-8
        inv = dist.pow(-self.n_exp)
        return inv / inv.sum(1, keepdim=True)

class HarmonicLoss(nn.Module):
    def forward(self, p, t):
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()

def run_epoch(enc, head, is_h, loader, crit, opt, dev):
    enc.train(bool(opt)); head.train(bool(opt))
    total = 0
    loss_sum = 0.0
    correct = 0
    for X, y in loader:
        X, y = X.to(dev), y.to(dev)
        if opt: opt.zero_grad()
        h = enc(X)
        if is_h:
            p = head(h)
            loss = crit(p, y)
            pred = p.argmax(1)
        else:
            logits = head(h)
            loss = crit(logits, y)
            pred = logits.argmax(1)
        if opt:
            loss.backward()
            opt.step()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        correct += (pred == y).sum().item()
    return loss_sum / total, correct / total


# ─── single‐run anomaly‐detection ─────────────────────────────────────

def one_run(dataset, held_out, seed, args):
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── load train/test ───────────────────────────────
    if args.data_dir:
        dd = pathlib.Path(args.data_dir)
        Xtr, ytr = load_ucr_txt(dd / f"{dataset}_TRAIN.txt")
        Xte, yte = load_ucr_txt(dd / f"{dataset}_TEST.txt")
    elif dataset.lower() in [
        "bump3", "sine_freq", "step_pos", "square_duty", "chirp", "dual_tone", "motif"
    ]:
        Xtr, ytr = make_synthetic_dataset(dataset.lower(), seed=seed)
        Xte, yte = make_synthetic_dataset(dataset.lower(), seed=seed + 999)
    else:
        Xtr, ytr = load_dataset_aeon(dataset, "train")
        Xte, yte = load_dataset_aeon(dataset, "test")

    # ── hold‐out splitting ─────────────────────────────
    novel = held_out
    # training on non‐novel
    idx_known = np.where(ytr != novel)[0]
    Xk_tr, yk_tr = Xtr[idx_known], ytr[idx_known]
    # remap known labels to 0…K-1
    uniq = sorted(set(yk_tr))
    mapper = {v: i for i, v in enumerate(uniq)}
    yk_tr = np.vectorize(mapper.get)(yk_tr)
    K = len(uniq)

    # test labels → binary
    y_test_bin = (yte == novel).astype(int)

    # ── build loader ─────────────────────────────────
    ds = TensorDataset(torch.tensor(Xk_tr), torch.tensor(yk_tr))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # ── instantiate models & optimizers ─────────────
    enc_h = Encoder1D(64).to(dev)
    head_h = HarmaxHead(64, K, args.n_exp).to(dev)
    opt_h = torch.optim.AdamW(
        list(enc_h.parameters()) + list(head_h.parameters()),
        lr=args.lr, weight_decay=0.0
    )
    crit_h = HarmonicLoss()

    enc_s = Encoder1D(64).to(dev)
    head_s = SoftmaxHead(64, K).to(dev)
    opt_s = torch.optim.AdamW(
        list(enc_s.parameters()) + list(head_s.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    crit_s = nn.CrossEntropyLoss()

    # ── training loop ────────────────────────────────
    logging.info(f"[{dataset}] seed={seed}  training on {K} known classes…")
    for ep in range(1, args.epochs + 1):
        run_epoch(enc_h, head_h, True, dl, crit_h, opt_h, dev)
        run_epoch(enc_s, head_s, False, dl, crit_s, opt_s, dev)
        if ep % 50 == 0:
            _, acc_h = run_epoch(enc_h, head_h, True, dl, crit_h, None, dev)
            _, acc_s = run_epoch(enc_s, head_s, False, dl, crit_s, None, dev)
            logging.info(f" Ep {ep:3d} train‐acc HarMax={acc_h:.3f} Softmax={acc_s:.3f}")

    # ── scoring ───────────────────────────────────────
    enc_h.eval(); head_h.eval()
    enc_s.eval(); head_s.eval()
    with torch.no_grad():
        H = enc_h(torch.tensor(Xte).to(dev))
        P_h = head_h(H).cpu().numpy()
        S = enc_s(torch.tensor(Xte).to(dev))
        logits = head_s(S).cpu().numpy()

    # anomaly score = 1 − max class prob
    score_h = 1 - P_h.max(axis=1)
    probs_s = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    score_s = 1 - probs_s.max(axis=1)

    # metrics
    auc_h = roc_auc_score(y_test_bin, score_h)
    ap_h = average_precision_score(y_test_bin, score_h)
    auc_s = roc_auc_score(y_test_bin, score_s)
    ap_s = average_precision_score(y_test_bin, score_s)

    # ── save JSON & ROC plot ─────────────────────────
    out_res = pathlib.Path("results") / "anomaly_detection"
    out_res.mkdir(parents=True, exist_ok=True)
    js = {
        "dataset": dataset,
        "held_out": novel,
        "seed": seed,
        "roc_auc": {"harmax": auc_h, "softmax": auc_s},
        "avg_prec": {"harmax": ap_h, "softmax": ap_s}
    }
    fn_js = out_res / f"{dataset}_held{novel}_seed{seed}.json"
    json.dump(js, open(fn_js, "w"), indent=2)

    # ROC plot
    out_fig = pathlib.Path("figures") / "anomaly_detection"
    out_fig.mkdir(parents=True, exist_ok=True)
    fpr_h, tpr_h, _ = roc_curve(y_test_bin, score_h)
    fpr_s, tpr_s, _ = roc_curve(y_test_bin, score_s)
    plt.figure()
    plt.plot(fpr_h, tpr_h, label=f"HarMax (AUC={auc_h:.3f})")
    plt.plot(fpr_s, tpr_s, label=f"Softmax (AUC={auc_s:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{dataset} held‐out={novel}")
    plt.legend(loc="lower right")
    plt.savefig(out_fig / f"{dataset}_held{novel}_seed{seed}_roc.png", dpi=120)
    plt.close()

    return js


# ─── aggregator & driver ────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='',
                        help="point to local UCR txt files if aeon missing")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_exp', type=float, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--datasets', nargs='+', default=[
    "Adiac",
    "ArrowHead",
    "Car",
    "DistalPhalanxOutlineAgeGroup",
    "ElectricDevices",
    "FaceAll",
    "Fish",
    "InsectWingbeatSound",
    'ECG5000',
    "TwoPatterns",
    "UWaveGestureLibraryAll",
])
    parser.add_argument('--held_out', type=int, default=None,
                        help="class index to hold out as anomaly (defaults to minority class)")

    args = parser.parse_args()

    # collect per‐run JSONs
    all_results = []
    for ds in args.datasets:
        # load train to determine held_out if not specified
        if args.data_dir:
            Xtr, ytr = load_ucr_txt(pathlib.Path(args.data_dir) / f"{ds}_TRAIN.txt")
        elif ds.lower() in ["bump3", "sine_freq", "step_pos", "square_duty", "chirp", "dual_tone", "motif"]:
            Xtr, ytr = make_synthetic_dataset(ds.lower(), seed=0)
        else:
            Xtr, ytr = load_dataset_aeon(ds, "train")
        # choose held_out = label with minimal count if not specified
        if args.held_out is None:
            vals, counts = np.unique(ytr, return_counts=True)
            held_out = int(vals[np.argmin(counts)])
        else:
            held_out = args.held_out
            if held_out not in np.unique(ytr):
                logging.warning(f"Specified held_out={held_out} not in dataset {ds}, using minority class instead")
                vals, counts = np.unique(ytr, return_counts=True)
                held_out = int(vals[np.argmin(counts)])

        # load test data after determining held_out
        if args.data_dir:
            Xte, yte = load_ucr_txt(pathlib.Path(args.data_dir) / f"{ds}_TEST.txt")
        elif ds.lower() in ["bump3", "sine_freq", "step_pos", "square_duty", "chirp", "dual_tone", "motif"]:
            Xte, yte = make_synthetic_dataset(ds.lower(), seed=0 + 999)
        else:
            Xte, yte = load_dataset_aeon(ds, "test")

        # remap labels to 0…C-1 consistently
        all_labels = sorted(set(np.concatenate([ytr, yte])))
        lab_map = {lab: i for i, lab in enumerate(all_labels)}
        ytr = np.vectorize(lab_map.get)(ytr).astype(np.int64)
        yte = np.vectorize(lab_map.get)(yte).astype(np.int64)
        held_out = lab_map[held_out]  # Adjust held_out to new label mapping

        for seed in args.seeds:
            js = one_run(ds, held_out, seed, args)
            all_results.append(js)

    # ── aggregate by dataset ─────────────────────────────
    summary = []
    for ds in args.datasets:
        sub = [r for r in all_results if r["dataset"] == ds]
        har_aucs = [r["roc_auc"]["harmax"] for r in sub]
        sof_aucs = [r["roc_auc"]["softmax"] for r in sub]
        har_aps = [r["avg_prec"]["harmax"] for r in sub]
        sof_aps = [r["avg_prec"]["softmax"] for r in sub]
        summary.append({
            "dataset": ds,
            "har_auc_mu": np.mean(har_aucs),
            "har_auc_sd": np.std(har_aucs),
            "sof_auc_mu": np.mean(sof_aucs),
            "sof_auc_sd": np.std(sof_aucs),
            "har_ap_mu": np.mean(har_aps),
            "har_ap_sd": np.std(har_aps),
            "sof_ap_mu": np.mean(sof_aps),
            "sof_ap_sd": np.std(sof_aps),
        })

    # ── save summary JSON ────────────────────────────────
    out_sum = pathlib.Path("results") / "anomaly_detection"
    out_sum.mkdir(exist_ok=True, parents=True)
    json.dump(summary, open(out_sum / "summary.json", "w"), indent=2)

    # ── emit LaTeX table ─────────────────────────────────
    tex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Anomaly‐detection ROC‐AUC (mean$\pm$std) holding out the rarest class.}",
        r"\label{tab:anomaly}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Dataset & HarMax & Softmax \\",
        r"\midrule"
    ]
    for s in summary:
        tex.append(
            f"{s['dataset']} & "
            f"${s['har_auc_mu']:.3f}\\pm{s['har_auc_sd']:.3f}$ & "
            f"${s['sof_auc_mu']:.3f}\\pm{s['sof_auc_sd']:.3f}$ \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(out_sum / "summary.tex", "w") as f:
        f.write("\n".join(tex))

    # also print to stdout
    print("\n".join(tex))


if __name__ == "__main__":
    main()