#!/usr/bin/env python3
"""
k-shot extension study for HarMax vs Softmax on UCR (via aeon) or synthetic datasets.
Holds out one class as “novel,” trains on the rest, then for various k adds k support
examples of the novel class and measures HarMax (via updated centre) vs Softmax (via
frozen encoder + logistic regression on embeddings).
"""
import argparse
import pathlib
import random
import logging
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# optional aeon loader for UCR
try:
    from aeon.datasets import load_classification
except ImportError:
    load_classification = None

# synthetic dispatcher (for bump3, sine_freq, etc.)
from utils.utils_synthetic_datasets import make_synthetic_dataset

# ─── model components ─────────────────────────────────────────────
class Encoder1D(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1))

class SoftmaxHead(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.logits = nn.Linear(d, k)
    def forward(self, h):
        return self.logits(h)

class HarmaxHead(nn.Module):
    def __init__(self, d, k, n_exp=None):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(k, d) / np.sqrt(d))
        self.n_exp   = int(n_exp) if n_exp else int(np.sqrt(d))
    def forward(self, h):
        dist = torch.cdist(h, self.centres) + 1e-8
        inv  = dist.pow(-self.n_exp)
        return inv / inv.sum(1, keepdim=True)

class HarmonicLoss(nn.Module):
    def forward(self, p, t):
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()

# ─── data loader ──────────────────────────────────────────────────
def load_dataset_aeon(name: str, split: str="train") -> Tuple[np.ndarray, np.ndarray]:
    if load_classification is None:
        raise RuntimeError("aeon not installed; cannot load UCR via aeon")
    try:
        X, y = load_classification(name, split=split, return_type="numpy3d")
    except TypeError:
        X, y = load_classification(name, split=split)
    X = np.asarray(X)
    if X.ndim == 3:
        X = X[:, 0, :]
    return X.astype("float32"), np.asarray(y, dtype="int64")

# ─── training utility ─────────────────────────────────────────────
def run_epoch(enc, head, is_h, loader, crit, opt, dev):
    enc.train(bool(opt)); head.train(bool(opt))
    total = correct = 0
    loss_sum = 0.0
    for X, y in loader:
        X, y = X.to(dev), y.to(dev)
        if opt: opt.zero_grad()
        h = enc(X)
        if is_h:
            p    = head(h); loss = crit(p, y); pred = p.argmax(1)
        else:
            logits = head(h); loss = crit(logits, y); pred = logits.argmax(1)
        if opt:
            loss.backward(); opt.step()
        total   += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum+= loss.item() * y.size(0)
    return loss_sum/total, correct/total

# ─── few-shot extension ────────────────────────────────────────────
def few_shot_extension(
    enc_h: Encoder1D, head_h: HarmaxHead,
    enc_s: Encoder1D,
    S_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test:  np.ndarray,
    X_support: np.ndarray, shots:    List[int],
    device: torch.device
) -> List[Tuple[int,float,float]]:
    enc_h.eval(); enc_s.eval()
    with torch.no_grad():
        H_test       = enc_h(torch.tensor(X_test).to(device)).cpu().numpy()
        S_test       = enc_s(torch.tensor(X_test).to(device)).cpu().numpy()
        S_train_emb  = enc_s(torch.tensor(S_train).to(device)).cpu().numpy()

    results = []
    base_k      = head_h.centres.shape[0]
    max_support = X_support.shape[0]

    for k in shots:
        n_sup = min(k, max_support)
        sup   = X_support[:n_sup]
        with torch.no_grad():
            emb_sup = enc_h(torch.tensor(sup).to(device)).cpu().numpy()

        # HarMax
        new_c    = emb_sup.mean(0, keepdims=True)
        centres  = np.vstack([head_h.centres.detach().cpu().numpy(), new_c])
        dmat     = np.linalg.norm(H_test[:,None,:] - centres[None,:,:], axis=2) + 1e-8
        inv      = dmat**(-head_h.n_exp)
        p_h      = inv / inv.sum(1, keepdims=True)
        acc_h    = (p_h.argmax(1) == y_test).mean()

        # Softmax few-shot via logistic regression
        X_lr = np.vstack([S_train_emb, emb_sup])
        y_lr = np.concatenate([y_train, np.full(n_sup, base_k)])
        lr   = LogisticRegression(multi_class='multinomial',
                                  solver='lbfgs', max_iter=500)
        lr.fit(X_lr, y_lr)
        acc_s = accuracy_score(y_test, lr.predict(S_test))

        results.append((n_sup, acc_h, acc_s))

    return results

# ─── main entrypoint ───────────────────────────────────────────────
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',  default='FaceFour')
    p.add_argument('--data_dir', default='')
    p.add_argument('--held_out', type=int,   default=0)
    p.add_argument('--shots',    nargs='+',  type=int, default=[1,5,10,20,50])
    p.add_argument('--epochs',   type=int,   default=20)
    p.add_argument('--batch',    type=int,   default=32)
    p.add_argument('--lr',       type=float, default=3e-3)
    p.add_argument('--wd',       type=float, default=0)
    p.add_argument('--device',   default='cuda')
    p.add_argument('--n_exp',    type=float, default=None)
    args = p.parse_args()

    # reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── load train/test ────────────────────────────────
    if args.data_dir:
        td   = pathlib.Path(args.data_dir)
        X_tr = np.load(td/f"{args.dataset}_TRAIN.npy")
        y_tr = np.load(td/f"{args.dataset}_TRAIN_labels.npy")
        X_te = np.load(td/f"{args.dataset}_TEST.npy")
        y_te = np.load(td/f"{args.dataset}_TEST_labels.npy")
    else:
        X_tr, y_tr = load_dataset_aeon(args.dataset, "train")
        X_te, y_te = load_dataset_aeon(args.dataset, "test")

    # ─── remap raw labels → 0…C-1 ───────────────────────────
    uniq = {lab: i for i, lab in enumerate(sorted(set(y_tr)))}
    y_tr = np.vectorize(uniq.get)(y_tr)
    y_te = np.vectorize(uniq.get)(y_te)

    # z-score each series
    X_tr = zscore(X_tr, axis=1)
    X_te = zscore(X_te, axis=1)

    # hold out one class as “novel”
    novel    = args.held_out
    sup_idx  = np.where(y_tr == novel)[0]
    part_idx = np.where(y_tr != novel)[0]
    X_support= X_tr[sup_idx]

    # reindex remaining training labels to 0…C-2
    y_part = y_tr[part_idx].copy()
    y_part[y_part > novel] -= 1
    X_part = X_tr[part_idx]

    ds_part = TensorDataset(torch.tensor(X_part), torch.tensor(y_part))
    dl_part = DataLoader(ds_part, batch_size=args.batch, shuffle=True)

    # instantiate models
    n_known = len(np.unique(y_part))
    enc_h   = Encoder1D(64).to(device)
    head_h  = HarmaxHead(64, n_known, args.n_exp).to(device)
    opt_h   = torch.optim.AdamW(list(enc_h.parameters())+list(head_h.parameters()),
                                lr=args.lr, weight_decay=0.0)
    crit_h  = HarmonicLoss()

    enc_s   = Encoder1D(64).to(device)
    head_s  = SoftmaxHead(64, n_known).to(device)
    opt_s   = torch.optim.AdamW(list(enc_s.parameters())+list(head_s.parameters()),
                                lr=args.lr, weight_decay=args.wd)
    crit_s  = nn.CrossEntropyLoss()

    # train on known classes
    logging.info("Training HarMax on known classes…")
    for ep in range(1, args.epochs+1):
        _, acc = run_epoch(enc_h, head_h, True, dl_part, crit_h, opt_h, device)
        if ep % 50 == 0:
            logging.info(f"  HarMax epoch {ep} → {acc*100:.1f}%")

    logging.info("Training Softmax on known classes…")
    for ep in range(1, args.epochs+1):
        _, acc = run_epoch(enc_s, head_s, False, dl_part, crit_s, opt_s, device)
        if ep % 50 == 0:
            logging.info(f"  Softmax epoch {ep} → {acc*100:.1f}%")

    # reindex test labels for novel
    y_test = y_te.copy()
    y_test[y_test > novel] -= 1

    # few-shot extension
    results = few_shot_extension(
        enc_h, head_h, enc_s,
        X_part, y_part,
        X_te,   y_test,
        X_support,
        shots=args.shots,
        device=device
    )

    print("\n  k-shot   HarMax   Softmax")
    for k, ah, as_ in results:
        print(f"{k:>6d}   {ah*100:6.2f}%   {as_*100:6.2f}%")
