#!/usr/bin/env python3
"""
Anomaly-detection via HarMax vs Softmax on UCR (or synthetic) series.

– hold out one class as “anomaly”
– train on remaining classes (early-stopping supported)
– score test points by 1 − max predicted probability
– compute ROC-AUC & Average-Precision
– save JSON & ROC plot
"""
import argparse, pathlib, random, json, logging, math, time, torch
from typing import Tuple

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

# optional aeon loader ------------------------------------------------
try:
    from aeon.datasets import load_classification
except ImportError:
    load_classification = None

# synthetic data ------------------------------------------------------
from utils.utils_synthetic_datasets import make_synthetic_dataset

# --------------------------------------------------------------------#
def load_ucr_txt(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Classic UCR .txt loader"""
    series, labels = [], []
    with open(path) as f:
        for ln in f:
            parts = ln.strip().replace("\t", " ").split(',')
            if len(parts) == 1:               # white-space separated
                parts = ln.split()
            lab, *vals = parts
            labels.append(float(lab))
            series.append([float(v) for v in vals])
    X = np.asarray(series, dtype=np.float32)
    y_raw = np.asarray(labels, dtype=np.int64)
    uniq = {lab: i for i, lab in enumerate(sorted(set(y_raw)))}
    y = np.vectorize(uniq.get)(y_raw).astype(np.int64)
    return X, y


def load_dataset_aeon(name: str, split: str = "train"):
    if load_classification is None:
        raise RuntimeError("aeon not installed – cannot load via aeon")
    try:
        X, y = load_classification(name, split=split, return_type="numpy3d")
    except TypeError:                          # older versions
        X, y = load_classification(name, split=split)
    X = np.asarray(X)
    if X.ndim == 3:                            # (N, 1, L)
        X = X[:, 0, :]
    return X.astype(np.float32), y.astype(np.int64)


# ----------------------------- model bits --------------------------- #
class Encoder1D(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, d), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))        # (B, L) → (B, d)


class SoftmaxHead(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.logits = nn.Linear(d, k)

    def forward(self, h):
        return self.logits(h)


class HarmaxHead(nn.Module):
    def __init__(self, d: int, k: int, n_exp=None):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(k, d) / math.sqrt(d))
        self.n_exp = int(n_exp) if n_exp else int(math.sqrt(d))

    def forward(self, h):
        dist = torch.cdist(h, self.centres) + 1e-8
        inv = dist.pow(-self.n_exp)
        return inv / inv.sum(1, keepdim=True)


class HarmonicLoss(nn.Module):
    def forward(self, p, t):
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()


@torch.no_grad()
def accuracy(enc, head, is_h, loader, dev):
    enc.eval(); head.eval()
    tot = cor = 0
    for X, y in loader:
        h = enc(X.to(dev))
        probs_or_logits = head(h)
        pred = probs_or_logits.argmax(1)
        tot += y.size(0)
        cor += (pred.cpu() == y).sum().item()
    return cor / tot


def run_epoch(enc, head, is_h, loader, crit, opt, dev):
    enc.train(bool(opt)); head.train(bool(opt))
    for X, y in loader:
        X, y = X.to(dev), y.to(dev)
        if opt:
            opt.zero_grad()
        h = enc(X)
        out = head(h)
        loss = crit(out, y)
        if opt:
            loss.backward()
            opt.step()


# ----------------------------- main -------------------------------- #
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", default="ECG5000")
    pa.add_argument("--data_dir", default="")
    pa.add_argument("--held_out", type=int, default=0,
                    help="class index to treat as anomaly")
    pa.add_argument("--epochs", type=int, default=200)
    pa.add_argument("--batch", type=int, default=32)
    pa.add_argument("--lr", type=float, default=3e-3)
    pa.add_argument("--wd", type=float, default=0.0)
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--n_exp", type=float, default=None)
    # early-stopping knobs
    pa.add_argument("--patience", type=int, default=20)
    pa.add_argument("--min_delta", type=float, default=1e-4)
    args = pa.parse_args()

    # ------------------- seeding & device ---------------------------
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------- load train / test -------------------------
    if args.data_dir:
        base = pathlib.Path(args.data_dir)
        Xtr, ytr = load_ucr_txt(base / f"{args.dataset}_TRAIN.txt")
        Xte, yte = load_ucr_txt(base / f"{args.dataset}_TEST.txt")
    elif args.dataset.lower() in ["bump3", "sine_freq", "step_pos",
                                  "square_duty", "chirp", "dual_tone", "motif"]:
        Xtr, ytr = make_synthetic_dataset(args.dataset.lower(), seed=0)
        Xte, yte = make_synthetic_dataset(args.dataset.lower(), seed=1)
    else:
        Xtr, ytr = load_dataset_aeon(args.dataset, "train")
        Xte, yte = load_dataset_aeon(args.dataset, "test")

    # relabel 0…C-1 consistently
    all_labels = sorted(set(ytr))
    lab_map = {lab: i for i, lab in enumerate(all_labels)}
    ytr = np.vectorize(lab_map.get)(ytr)
    yte = np.vectorize(lab_map.get)(yte)
    C = len(all_labels)

    novel = args.held_out
    if novel < 0 or novel >= C:
        raise ValueError(f"--held_out must be in 0…{C-1}")

    # -------------- create train loader (known classes) ------------
    keep = ytr != novel
    Xk_tr, yk_tr = Xtr[keep], ytr[keep]
    remap = {lab: i for i, lab in enumerate(sorted(set(yk_tr)))}
    yk_tr = np.vectorize(remap.get)(yk_tr)
    K = len(remap)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(Xk_tr), torch.tensor(yk_tr)),
        batch_size=args.batch, shuffle=True)

    # -------------- test labels (binary) ----------------------------
    y_test_bin = (yte == novel).astype(int)
    if y_test_bin.min() == y_test_bin.max():
        logging.warning("Test split lacks anomalies for class %d – aborting.", novel)
        return

    # -------------- build models -----------------------------------
    enc_h = Encoder1D(64).to(dev)
    head_h = HarmaxHead(64, K, args.n_exp).to(dev)
    enc_s = Encoder1D(64).to(dev)
    head_s = SoftmaxHead(64, K).to(dev)

    opt_h = torch.optim.AdamW(
        list(enc_h.parameters()) + list(head_h.parameters()),
        lr=args.lr)
    opt_s = torch.optim.AdamW(
        list(enc_s.parameters()) + list(head_s.parameters()),
        lr=args.lr, weight_decay=args.wd)
    crit_h = HarmonicLoss()
    crit_s = nn.CrossEntropyLoss()

    # -------------- early-stopping logic ---------------------------
    best_acc = 0.0
    best_state = None
    wait = 0
    start = time.time()

    logging.info("Training on %d known classes with patience=%d…", K, args.patience)
    for ep in range(1, args.epochs + 1):
        run_epoch(enc_h, head_h, True, train_dl, crit_h, opt_h, dev)
        run_epoch(enc_s, head_s, False, train_dl, crit_s, opt_s, dev)

        # evaluate HarMax train accuracy
        acc_h = accuracy(enc_h, head_h, True, train_dl, dev)

        if acc_h > best_acc + args.min_delta:
            best_acc = acc_h
            best_state = (enc_h.state_dict(), head_h.state_dict(),
                          enc_s.state_dict(), head_s.state_dict())
            wait = 0
        else:
            wait += 1

        if ep % 50 == 0 or ep == 1:
            acc_s = accuracy(enc_s, head_s, False, train_dl, dev)
            logging.info("Ep %3d  acc_H=%.3f  acc_S=%.3f  (wait=%d)", ep, acc_h, acc_s, wait)

        if wait >= args.patience:
            logging.info("Early stopped at epoch %d (no improv %d×)", ep, args.patience)
            break

    # restore best weights -------------------------------------------------
    if best_state is not None:
        enc_h.load_state_dict(best_state[0])
        head_h.load_state_dict(best_state[1])
        enc_s.load_state_dict(best_state[2])
        head_s.load_state_dict(best_state[3])

    # -------------- scoring ----------------------------------------------
    enc_h.eval(); head_h.eval()
    enc_s.eval(); head_s.eval()
    with torch.no_grad():
        H = enc_h(torch.tensor(Xte).to(dev)).cpu()
        probs_h = head_h(H).cpu().numpy()
        S = enc_s(torch.tensor(Xte).to(dev)).cpu()
        logits_s = head_s(S).cpu().numpy()

    score_h = 1.0 - probs_h.max(axis=1)
    prob_s = np.exp(logits_s) / np.exp(logits_s).sum(axis=1, keepdims=True)
    score_s = 1.0 - prob_s.max(axis=1)

    auc_h = roc_auc_score(y_test_bin, score_h)
    ap_h = average_precision_score(y_test_bin, score_h)
    auc_s = roc_auc_score(y_test_bin, score_s)
    ap_s = average_precision_score(y_test_bin, score_s)

    print(f"\nResults on `{args.dataset}`, held-out={novel}")
    print(f" HarMax  AUC={auc_h:.3f}  AP={ap_h:.3f}")
    print(f" Softmax AUC={auc_s:.3f}  AP={ap_s:.3f}")

    # ---------------- save JSON & ROC plot --------------------------
    out_res = pathlib.Path("results") / "anomaly"
    out_fig = pathlib.Path("figures") / "anomaly"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    res = dict(dataset=args.dataset, held_out=novel, n_known=K,
               roc_auc=dict(harmax=auc_h, softmax=auc_s),
               avg_prec=dict(harmax=ap_h, softmax=ap_s),
               epochs_trained=ep, best_train_acc=float(best_acc),
               elapsed_sec=int(time.time() - start))
    with (out_res / f"{args.dataset}_held{novel}.json").open("w") as f:
        json.dump(res, f, indent=2)

    fpr_h, tpr_h, _ = roc_curve(y_test_bin, score_h)
    fpr_s, tpr_s, _ = roc_curve(y_test_bin, score_s)
    fig, ax = plt.subplots()
    ax.plot(fpr_h, tpr_h, label=f"HarMax (AUC={auc_h:.3f})")
    ax.plot(fpr_s, tpr_s, label=f"Softmax (AUC={auc_s:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{args.dataset} anomaly ROC")
    ax.legend(loc="lower right")
    fig.savefig(out_fig / f"{args.dataset}_held{novel}_roc.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
