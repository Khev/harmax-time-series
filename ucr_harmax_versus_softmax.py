#!/usr/bin/env python3
# ---------------------------------------------------------------------
# soft-max  vs.  harmax   on any univariate UCR dataset
#   • trains both heads
#   • saves weights       → models/<dataset>/
#   • saves metrics JSON  → results/<dataset>_seed…json
#   • saves plots         → figures/
# ---------------------------------------------------------------------
import argparse, math, pathlib, random, json, warnings

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import zscore

from sklearn.decomposition import PCA        # ⟨NEW⟩ for 2-D visualisation
# -------------------------------------------------------------- imports
try:
    from aeon.datasets import load_classification          # optional
except ModuleNotFoundError:
    load_classification = None
    warnings.warn("aeon not found – will only load local .txt files")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------


# ------------------------------- synthetic “phase-wheel” -------------
def make_phase_wheel(
        n_classes=8, samples_per_class=400, length=128,
        freq=3.0, noise_std=0.15, seed=0):
    """
    Simple toy set: sine-waves with equally spaced phase-shifts.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length, endpoint=False)

    X, y = [], []
    for k in range(n_classes):
        phi = 2 * np.pi * k / n_classes
        centre = np.sin(2 * np.pi * freq * t + phi)
        amps = 1.0 + rng.normal(0, 0.05, size=samples_per_class)
        noise = rng.normal(0, noise_std, size=(samples_per_class, length))
        Xk = amps[:, None] * centre + noise
        X.append(Xk)
        y.append(np.full(samples_per_class, k))

    return np.vstack(X).astype("float32"), np.concatenate(y).astype("int64")


# ------------------------- misc helpers ------------------------------
def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_ucr_txt(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    series, labels = [], []
    with open(path) as f:
        for ln in f:
            parts = ln.replace('\t', ' ').strip().split(',')
            if len(parts) == 1:
                parts = ln.split()
            lab, *vals = parts
            labels.append(float(lab))
            series.append([float(v) for v in vals])
    X = np.asarray(series, np.float32)
    y_raw = np.asarray(labels, np.int64)
    uniq = {lab: i for i, lab in enumerate(sorted(set(y_raw)))}
    y = np.vectorize(uniq.get)(y_raw).astype("int64")
    return X, y


def load_dataset_aeon(name, split="train"):
    if load_classification is None:
        raise RuntimeError("aeon missing – install it or use --data_dir")
    try:
        X, y = load_classification(name, split=split, return_type="numpy3d")
    except TypeError:                       # older aeon
        X, y = load_classification(name, split=split)
    if isinstance(X, np.ndarray) and X.ndim == 3:
        X = X[:, 0, :]
    else:                                   # nested DF → array
        from aeon.utils import convert_series_to_array
        X = convert_series_to_array(X)
    return X.astype("float32"), y.astype("int64")


# ------------------------------ model bits ---------------------------
class Encoder1D(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, d), nn.ReLU()
        )

    def forward(self, x):                   # x: (B, L)
        return self.net(x.unsqueeze(1))


class SoftmaxHead(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.logits = nn.Linear(d, k)

    def forward(self, h):
        return self.logits(h)


class HarmaxHead(nn.Module):
    def __init__(self, d, k, n=None):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(k, d) / math.sqrt(d))
        self.n_exp = int(n) if n else int(math.sqrt(d))

    def probs(self, h):
        dist = torch.cdist(h, self.centres) + 1e-8
        inv = dist.pow(-self.n_exp)
        return inv / inv.sum(1, keepdim=True)

    def forward(self, h):
        return self.probs(h)


class HarmonicLoss(nn.Module):
    def forward(self, p, t):
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()


def run_epoch(model, is_h, loader, crit, opt, dev):
    model.train(bool(opt))
    tot = cor = loss = 0
    for X, y in loader:
        X, y = X.to(dev), y.to(dev)
        if opt: opt.zero_grad()
        h = model.encoder(X)
        if is_h:
            p = model.head(h); l = crit(p, y); pred = p.argmax(1)
        else:
            logit = model.head(h); l = crit(logit, y); pred = logit.argmax(1)
        if opt:
            l.backward(); opt.step()
        b = y.size(0)
        loss += l.item() * b; cor += (pred == y).sum().item(); tot += b
    return loss / tot, cor / tot


class Wrap(nn.Module):
    """Tiny container so we can keep `.is_harmax` flag together."""
    def __init__(self, enc, head, is_h):
        super().__init__()
        self.encoder, self.head, self.is_harmax = enc, head, is_h


# --------------------------- plotting helpers ------------------------
def plot_pca(models: dict, dataset: str, out_tag: str):
    """
    Make 2-D PCA scatter of the per-class vectors (weights or centres).
    """
    def vecs(mod):
        return (mod.head.centres if mod.is_harmax
                else mod.head.logits.weight).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for i, (name, mod) in enumerate(models.items()):
        X = vecs(mod)
        pcs = PCA(2).fit_transform(X)
        ev = PCA(2).fit(X).explained_variance_ratio_.sum() * 100
        ax[i].scatter(pcs[:, 0], pcs[:, 1], s=35)
        for j, (x, y) in enumerate(pcs):
            ax[i].text(x, y, str(j), ha='center', va='center', fontsize=7)
        ax[i].set_aspect('equal')
        ax[i].set_title(f"{name}  EV:{ev:0.0f}%")
    fig.tight_layout()
    p = pathlib.Path("figures")
    p.mkdir(exist_ok=True)
    fig.savefig(p / f"{dataset}_pca_{out_tag}.png", dpi=120)
    plt.close(fig)


# ------------------------------ main ---------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset', default='ECG200')
    pa.add_argument('--data_dir', default='')
    pa.add_argument('--epochs', type=int, default=400)
    pa.add_argument('--batch',  type=int, default=32)
    pa.add_argument('--lr',     type=float, default=3e-3)
    pa.add_argument('--wd',     type=float, default=1e-2)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pa.add_argument('--seed',   type=int, default=0)
    pa.add_argument('--n_exp',  type=float, default=None,
                    help='harmonic exponent n (default √d)')
    pa.add_argument('--no_plot', action='store_true')
    args = pa.parse_args()

    set_seed(args.seed)
    dev = torch.device(args.device)

    # ------------ load data ---------------------------
    if args.data_dir:
        dd = pathlib.Path(args.data_dir)
        Xtr, ytr = load_ucr_txt(dd / f"{args.dataset}_TRAIN.txt")
        Xte, yte = load_ucr_txt(dd / f"{args.dataset}_TEST.txt")
    elif args.dataset.lower() == "phasewheel":
        Xtr, ytr = make_phase_wheel(n_classes=8, samples_per_class=300, seed=args.seed)
        Xte, yte = make_phase_wheel(n_classes=8, samples_per_class=300, seed=999 + args.seed)
    else:
        Xtr, ytr = load_dataset_aeon(args.dataset, "train")
        Xte, yte = load_dataset_aeon(args.dataset, "test")

    # remap labels → 0…C-1
    uniq = {lab: i for i, lab in enumerate(sorted(set(ytr)))}
    ytr = np.vectorize(uniq.get)(ytr); yte = np.vectorize(uniq.get)(yte)
    n_cls = len(uniq)

    # per-series z-score
    Xtr, Xte = zscore(Xtr, axis=1), zscore(Xte, axis=1)
    print(f"{args.dataset}: {Xtr.shape} → {n_cls} classes")

    tr_dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                       batch_size=args.batch, shuffle=True, drop_last=False)
    te_dl = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
                       batch_size=args.batch, drop_last=False)

    models = {
        "softmax": Wrap(Encoder1D(64), SoftmaxHead(64, n_cls), False),
        "harmax" : Wrap(Encoder1D(64), HarmaxHead(64, n_cls, args.n_exp), True)
    }
    crit = {"softmax": nn.CrossEntropyLoss(), "harmax": HarmonicLoss()}
    opt  = {n: torch.optim.AdamW(m.parameters(), lr=args.lr,
                                 weight_decay=args.wd if n == "softmax" else 0.0)
            for n, m in models.items()}
    hist = {n: {"tr": [], "te": []} for n in models}

    # ------------------ training loop -----------------
    for ep in range(1, args.epochs + 1):
        for n, m in models.items():
            m.to(dev)
            tr_l, tr_a = run_epoch(m, m.is_harmax, tr_dl, crit[n], opt[n], dev)
            te_l, te_a = run_epoch(m, m.is_harmax, te_dl, crit[n], None, dev)
            hist[n]['tr'].append(tr_a); hist[n]['te'].append(te_a)
        if ep % 50 == 0 or ep == 1:
            print(f"Ep {ep:3d}  " +
                  " ".join(f"{n}:{hist[n]['te'][-1]*100:.1f}%" for n in models))

    har_n = models['harmax'].head.n_exp

    # ------------------ plots -------------------------
    if not args.no_plot:
        xs = np.arange(1, args.epochs + 1)
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        for i, name in enumerate(["softmax", "harmax"]):
            ax[i].plot(xs, hist[name]['tr'], c='tab:blue', label='train')
            ax[i].plot(xs, hist[name]['te'], c='tab:orange', label='test')
            ax[i].set_ylim(0, 1.05); ax[i].set_title(name); ax[i].legend()
        ax[1].set_xlabel("Epoch")
        pathlib.Path("figures").mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(f"figures/{args.dataset}_acc_n{har_n}.png", dpi=120)
        plt.close(fig)

        # ⟨NEW⟩ 2-D PCA of the class vectors
        plot_pca(models, args.dataset, f"n{har_n}")

    # ------------------ save weights ------------------
    mdl_dir = pathlib.Path("models") / args.dataset
    mdl_dir.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        torch.save(m.state_dict(),
                   mdl_dir / f"{name}_seed{args.seed}_n{har_n}.pth")

    # ------------------ save metrics ------------------
    half = args.epochs // 2
    metrics = {
        "dataset": args.dataset, "seed": args.seed, "n_exp": har_n,
        "soft_hist_tr": hist['softmax']['tr'],
        "soft_hist_te": hist['softmax']['te'],
        "har_hist_tr":  hist['harmax']['tr'],
        "har_hist_te":  hist['harmax']['te'],
        "soft_last_te": float(hist['softmax']['te'][-1]),
        "har_last_te":  float(hist['harmax']['te'][-1]),
        "soft_mean_te": float(np.mean(hist['softmax']['te'][half:])),
        "har_mean_te":  float(np.mean(hist['harmax']['te'][half:])),
    }
    res_dir = pathlib.Path("results"); res_dir.mkdir(exist_ok=True)
    with (res_dir / f"{args.dataset}_seed{args.seed}_n{har_n}.json").open("w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
