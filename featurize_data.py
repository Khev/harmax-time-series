#!/usr/bin/env python3
# ---------------------------------------------------------------------
# soft-max vs. harmax on any univariate UCR dataset (or Catch22‐featurized)
#   • trains both heads
#   • saves weights       → models[/(featureize)]/<dataset>/
#   • saves metrics JSON  → results[/(featureize)]/<dataset>_…json
#   • saves plots         → figures[/(featureize)]/
#   • supports early stopping
# ---------------------------------------------------------------------
import argparse
import math
import pathlib
import random
import json
import warnings
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import zscore
from sklearn.decomposition import PCA        # for 2-D visualisation

# Catch22 feature extractor
try:
    from pycatch22 import catch22_all
except ImportError:
    catch22_all = None

# force Agg backend so this works in headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# our plotting utilities
from utils.utils_plotting import (
    get_embeddings,
    plot_embeddings_comparison,
    plot_prototypes
)
# synthetic data generator
from utils.utils_synthetic_datasets import make_synthetic_dataset

# optional aeon loader
try:
    from aeon.datasets import load_classification
except ModuleNotFoundError:
    load_classification = None
    warnings.warn("aeon not found – will only load local .txt files")


# ── misc helpers ─────────────────────────────────────────────────────
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ucr_txt(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    series, labels = [], []
    with open(path) as f:
        for ln in f:
            parts = ln.replace('\t',' ').strip().split(',')
            if len(parts)==1:
                parts = ln.split()
            lab, *vals = parts
            labels.append(float(lab))
            series.append([float(v) for v in vals])
    X = np.asarray(series, dtype=np.float32)
    y_raw = np.asarray(labels, dtype=np.int64)
    uniq = {lab:i for i,lab in enumerate(sorted(set(y_raw)))}
    y = np.vectorize(uniq.get)(y_raw).astype(np.int64)
    return X, y


def load_dataset_aeon(name, split="train"):
    if load_classification is None:
        raise RuntimeError("aeon missing – install it or use --data_dir")
    try:
        X, y = load_classification(name, split=split, return_type="numpy3d")
    except TypeError:
        X, y = load_classification(name, split=split)
    if isinstance(X, np.ndarray) and X.ndim==3:
        X = X[:,0,:]
    else:
        from aeon.utils import convert_series_to_array
        X = convert_series_to_array(X)
    return X.astype(np.float32), y.astype(np.int64)


# ── model bits ──────────────────────────────────────────────────────
class Encoder1D(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,64,7,padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,5,padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128,d), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1))   # x: (B,L) → (B,d)


class SoftmaxHead(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.logits = nn.Linear(d,k)
    def forward(self,h):
        return self.logits(h)


class HarmaxHead(nn.Module):
    def __init__(self, d, k, n_exp=None):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(k,d)/math.sqrt(d))
        self.n_exp   = int(n_exp) if n_exp else int(math.sqrt(d))
    def forward(self,h):
        dist = torch.cdist(h, self.centres) + 1e-8
        inv  = dist.pow(-self.n_exp)
        return inv / inv.sum(1,keepdim=True)


class HarmonicLoss(nn.Module):
    def forward(self,p,t):
        return -torch.log(p[torch.arange(p.size(0)),t]).mean()


def run_epoch(model, is_h, loader, crit, opt, dev):
    model.train(bool(opt))
    tot=cor=loss=0.0
    for X,y in loader:
        X,y = X.to(dev), y.to(dev)
        if opt: opt.zero_grad()
        h = model.encoder(X)
        if is_h:
            p = model.head(h); l = crit(p,y); pred = p.argmax(1)
        else:
            logits = model.head(h); l = crit(logits,y); pred = logits.argmax(1)
        if opt:
            l.backward(); opt.step()
        b = y.size(0)
        loss += l.item()*b
        cor  += (pred==y).sum().item()
        tot  += b
    return loss/tot, cor/tot


class Wrap(nn.Module):
    def __init__(self, enc, head, is_h):
        super().__init__()
        self.encoder, self.head, self.is_harmax = enc, head, is_h


# ── main ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',  default='ECG5000')
    p.add_argument('--data_dir', default='')
    p.add_argument('--featurize', action='store_true',
                   help="use Catch22 features instead of raw series")
    p.add_argument('--epochs',   type=int,   default=100)
    p.add_argument('--batch',    type=int,   default=32)
    p.add_argument('--lr',       type=float, default=3e-3)
    p.add_argument('--wd',       type=float, default=0.0)
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',     type=int,   default=0)
    p.add_argument('--n_exp',    type=float, default=None,
                   help="harmonic exponent n (default √d)")
    p.add_argument('--no_plot',  action='store_true')
    args = p.parse_args()

    set_seed(args.seed)
    dev = torch.device(args.device)

    mode_dir = 'featureize' if args.featurize else ''

    # ── load raw series ─────────────────────────────────────────────
    if args.data_dir:
        dd = pathlib.Path(args.data_dir)
        Xtr, ytr = load_ucr_txt(dd/f"{args.dataset}_TRAIN.txt")
        Xte, yte = load_ucr_txt(dd/f"{args.dataset}_TEST.txt")
    elif args.dataset.lower() in ["bump3","sine_freq","step_pos",
                                   "square_duty","chirp","dual_tone","motif"]:
        Xtr, ytr = make_synthetic_dataset(args.dataset.lower(), seed=args.seed)
        Xte, yte = make_synthetic_dataset(args.dataset.lower(), seed=999+args.seed)
    else:
        Xtr, ytr = load_dataset_aeon(args.dataset, "train")
        Xte, yte = load_dataset_aeon(args.dataset, "test")

    # ── optionally featurize ─────────────────────────────────────────
    if args.featurize:
        if catch22_all is None:
            raise RuntimeError("Please `pip install pycatch22` to use --featurize")
        # compute 22 features per series (use the 'values' field)
        Xtr = np.vstack([catch22_all(x)['values'] for x in Xtr]).astype(np.float32)
        Xte = np.vstack([catch22_all(x)['values'] for x in Xte]).astype(np.float32)
        # per‐feature z-score
        Xtr = zscore(Xtr, axis=0)
        Xte = zscore(Xte, axis=0)
        Xtr = np.nan_to_num(Xtr, nan=0.0)
        Xte = np.nan_to_num(Xte, nan=0.0)

    # ── remap labels → 0..C-1 ────────────────────────────────────────
    uniq = {lab:i for i,lab in enumerate(sorted(set(ytr)))}
    ytr = np.vectorize(uniq.get)(ytr)
    yte = np.vectorize(uniq.get)(yte)
    n_cls = len(uniq)

    Xte_raw = Xte.copy()
    print(f"{args.dataset} [{mode_dir or 'raw'}]: {Xtr.shape} → {n_cls} classes")

    # ── build loaders ───────────────────────────────────────────────
    tr_dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                       batch_size=args.batch, shuffle=True)
    te_dl = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
                       batch_size=args.batch)

    # ── choose encoder ──────────────────────────────────────────────
    if args.featurize:
        enc = nn.Identity()       # features are already d-dimensional
        d   = Xtr.shape[1]
    else:
        enc = Encoder1D(64)
        d   = 64

    # ── instantiate models ──────────────────────────────────────────
    models = {
        "softmax": Wrap(enc, SoftmaxHead(d,n_cls), False),
        "harmax" : Wrap(enc, HarmaxHead(d,n_cls,args.n_exp), True)
    }
    crit = {
        "softmax": nn.CrossEntropyLoss(),
        "harmax":  HarmonicLoss()
    }
    opt = {
        name: torch.optim.AdamW(
            m.parameters(), lr=args.lr,
            weight_decay=(args.wd if name=="softmax" else 0.0)
        ) for name,m in models.items()
    }

    # ── training loop with early stopping ───────────────────────────
    hist = {n:{"tr":[], "te":[]} for n in models}
    patience=20; best_te=0.0; wait=0
    for ep in range(1, args.epochs+1):
        for name,m in models.items():
            m.to(dev)
            _,tr_a = run_epoch(m, m.is_harmax, tr_dl, crit[name], opt[name], dev)
            _,te_a = run_epoch(m, m.is_harmax, te_dl, crit[name], None,      dev)
            hist[name]["tr"].append(tr_a)
            hist[name]["te"].append(te_a)

        current = hist["harmax"]["te"][-1]
        if current > best_te+1e-6:
            best_te, wait = current, 0
        else:
            wait += 1

        if ep%50==0 or ep==1:
            logging.info(f"Ep {ep:3d}  softmax:{hist['softmax']['te'][-1]*100:.1f}%"
                         f"  harmax:{current*100:.1f}%")
        if wait>=patience:
            logging.info(f"Early stopping at ep {ep} (no improvement for {patience} epochs)")
            break
    else:
        ep = args.epochs

    har_n = models["harmax"].head.n_exp

    # ── plotting ───────────────────────────────────────────────────
    if not args.no_plot:
        base_fig = pathlib.Path("figures")/mode_dir
        base_fig.mkdir(parents=True, exist_ok=True)

        xs = np.arange(1, ep+1)
        fig,ax = plt.subplots(2,1,figsize=(6,6),sharex=True)
        for i,name in enumerate(["softmax","harmax"]):
            ax[i].plot(xs, hist[name]["tr"][:ep], label="train")
            ax[i].plot(xs, hist[name]["te"][:ep], label="test")
            ax[i].set_title(name); ax[i].set_ylim(0,1.05); ax[i].legend()
        ax[1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(base_fig/f"{args.dataset}_acc_feat{args.featurize}_n{har_n}.png", dpi=120)
        plt.close(fig)

        har_emb,har_lbls = get_embeddings(models["harmax"], te_dl, dev)
        soft_emb,_      = get_embeddings(models["softmax"], te_dl, dev)
        plot_embeddings_comparison(
            harmax_embeddings=har_emb,
            softmax_embeddings=soft_emb,
            labels=har_lbls,
            harmax_centres=models["harmax"].head.centres.detach().cpu().numpy(),
            softmax_weights=models["softmax"].head.logits.weight.detach().cpu().numpy(),
            save_path=base_fig/f"{args.dataset}_embeddings_feat{args.featurize}.png"
        )
        plot_prototypes(
            X_raw=Xte_raw,
            labels=har_lbls,
            harmax_embeddings=har_emb,
            harmax_centres=models["harmax"].head.centres.detach().cpu().numpy(),
            softmax_embeddings=soft_emb,
            softmax_weights=models["softmax"].head.logits.weight.detach().cpu().numpy(),
            save_path=base_fig/f"{args.dataset}_prototypes_feat{args.featurize}.png"
        )

    # ── save weights & metrics ───────────────────────────────────────
    base_mod = pathlib.Path("models")/mode_dir/args.dataset
    base_mod.mkdir(parents=True, exist_ok=True)
    for name,m in models.items():
        torch.save(m.state_dict(),
                   base_mod/f"{name}_seed{args.seed}_n{har_n}.pth")

    base_res = pathlib.Path("results")/mode_dir
    base_res.mkdir(parents=True, exist_ok=True)
    half = ep//2
    metrics = {
        "dataset":   args.dataset,
        "seed":      args.seed,
        "n_exp":     har_n,
        "featurize": args.featurize,
        "soft_last": hist["softmax"]["te"][ep-1],
        "har_last":  hist["harmax"]["te"][ep-1],
        "soft_mean": float(np.mean(hist["softmax"]["te"][half:ep])),
        "har_mean":  float(np.mean(hist["harmax"]["te"][half:ep])),
        "hist":      hist
    }
    with (base_res/f"{args.dataset}_seed{args.seed}_n{har_n}_feat{args.featurize}.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Finished")


if __name__=="__main__":
    main()
