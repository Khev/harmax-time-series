#!/usr/bin/env python3
"""
Anomaly‐detection via HarMax vs Softmax on UCR (or synthetic) series.

– hold out one class as “anomaly”
– train on remaining classes
– score test points by 1 − max predicted probability
– compute ROC-AUC & Average-Precision (if both classes present)
– save JSON & ROC plot
"""
import argparse, pathlib, random, json, logging, math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# optional aeon loader
try:
    from aeon.datasets import load_classification
except ImportError:
    load_classification = None

# synthetic data generator
from utils.utils_synthetic_datasets import make_synthetic_dataset

# ── data loaders ─────────────────────────────────────────────────────
def load_ucr_txt(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    series, labels = [], []
    with open(path) as f:
        for ln in f:
            parts = ln.strip().replace("\t"," ").split(',')
            if len(parts)==1: parts = ln.split()
            lab, *vals = parts
            labels.append(float(lab))
            series.append([float(v) for v in vals])
    X = np.array(series, dtype=np.float32)
    y_raw = np.array(labels, dtype=np.int64)
    # remap to 0…C-1
    uniq = {lab:i for i,lab in enumerate(sorted(set(y_raw)))}
    y = np.vectorize(uniq.get)(y_raw).astype(np.int64)
    return X, y

def load_dataset_aeon(name:str, split="train"):
    if load_classification is None:
        raise RuntimeError("aeon not installed – cannot load via aeon")
    try:
        X,y = load_classification(name, split=split, return_type="numpy3d")
    except TypeError:
        X,y = load_classification(name, split=split)
    X = np.asarray(X)
    if X.ndim==3: X = X[:,0,:]
    return X.astype(np.float32), y.astype(np.int64)


# ── model bits ───────────────────────────────────────────────────────
class Encoder1D(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,64,7,padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,5,padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, d), nn.ReLU()
        )
    def forward(self,x):
        return self.net(x.unsqueeze(1))

class SoftmaxHead(nn.Module):
    def __init__(self,d,k):
        super().__init__()
        self.logits = nn.Linear(d,k)
    def forward(self,h): return self.logits(h)

class HarmaxHead(nn.Module):
    def __init__(self,d,k,n_exp=None):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(k,d)/math.sqrt(d))
        self.n_exp = int(n_exp) if n_exp else int(math.sqrt(d))
    def forward(self,h):
        dist = torch.cdist(h, self.centres) + 1e-8
        inv  = dist.pow(-self.n_exp)
        return inv / inv.sum(1,keepdim=True)

class HarmonicLoss(nn.Module):
    def forward(self,p,t):
        return -torch.log(p[torch.arange(p.size(0)), t]).mean()

def run_epoch(enc, head, is_h, loader, crit, opt, dev):
    enc.train(bool(opt)); head.train(bool(opt))
    total=0; correct=0; loss_sum=0.0
    for X,y in loader:
        X,y = X.to(dev), y.to(dev)
        if opt: opt.zero_grad()
        h = enc(X)
        if is_h:
            p = head(h); loss = crit(p,y); pred = p.argmax(1)
        else:
            logits = head(h); loss = crit(logits,y); pred = logits.argmax(1)
        if opt:
            loss.backward(); opt.step()
        total   += y.size(0)
        correct += (pred==y).sum().item()
        loss_sum+= loss.item()*y.size(0)
    return loss_sum/total, correct/total


# ── main ─────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    p = argparse.ArgumentParser()
    p.add_argument('--dataset',  default='ECG5000')
    p.add_argument('--data_dir', default='')
    p.add_argument('--held_out', type=int, default=0,
                  help="index (0…C−1) of class to hold out as anomaly")
    p.add_argument('--epochs',   type=int, default=100)
    p.add_argument('--batch',    type=int, default=32)
    p.add_argument('--lr',       type=float, default=3e-3)
    p.add_argument('--wd',       type=float, default=0.0)
    p.add_argument('--device',   default='cuda')
    p.add_argument('--n_exp',    type=float, default=None)
    args = p.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── load data ───────────────────────────────────────
    if args.data_dir:
        td = pathlib.Path(args.data_dir)
        Xtr,ytr = load_ucr_txt(td/f"{args.dataset}_TRAIN.txt")
        Xte,yte = load_ucr_txt(td/f"{args.dataset}_TEST.txt")
    elif args.dataset.lower() in ["bump3","sine_freq","step_pos","square_duty","chirp","dual_tone","motif"]:
        Xtr,ytr = make_synthetic_dataset(args.dataset.lower(), seed=0)
        Xte,yte = make_synthetic_dataset(args.dataset.lower(), seed=1)
    else:
        Xtr,ytr = load_dataset_aeon(args.dataset,"train")
        Xte,yte = load_dataset_aeon(args.dataset,"test")

    # ── remap both train & test labels → 0…C−1 ───────────
    uniq = {lab:i for i,lab in enumerate(sorted(set(ytr)))}
    ytr = np.vectorize(uniq.get)(ytr)
    yte = np.vectorize(uniq.get)(yte)
    num_classes = len(uniq)

    # split off the held-out class
    novel = args.held_out
    if novel<0 or novel>=num_classes:
        raise ValueError(f"--held_out must be in [0,{num_classes-1}]")

    # known‐class training set
    known_mask = (ytr != novel)
    Xk_tr, yk_tr = Xtr[known_mask], ytr[known_mask]
    # remap known labels to 0…K−1
    known_labels = sorted(set(yk_tr))
    mapper = {v:i for i,v in enumerate(known_labels)}
    yk_tr = np.vectorize(mapper.get)(yk_tr)
    K = len(known_labels)

    # test set & binarize
    X_test       = Xte
    y_test_raw   = yte
    y_test_bin   = (yte == novel).astype(int)

    # ensure both classes present
    if not (0 in y_test_bin and 1 in y_test_bin):
        logging.warning("Test set does not contain both normal/anomaly for held_out=%d; skipping metrics", novel)
        return

    # ── build loader ─────────────────────────────────────
    ds = TensorDataset(torch.tensor(Xk_tr), torch.tensor(yk_tr))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    # ── instantiate & train ─────────────────────────────
    enc_h = Encoder1D(64).to(dev)
    head_h= HarmaxHead(64, K, args.n_exp).to(dev)
    opt_h = torch.optim.AdamW(list(enc_h.parameters()) + list(head_h.parameters()),
                              lr=args.lr, weight_decay=0.0)
    crit_h = HarmonicLoss()

    enc_s = Encoder1D(64).to(dev)
    head_s= SoftmaxHead(64, K).to(dev)
    opt_s = torch.optim.AdamW(list(enc_s.parameters()) + list(head_s.parameters()),
                              lr=args.lr, weight_decay=args.wd)
    crit_s = nn.CrossEntropyLoss()

    logging.info("Training on %d known classes…", K)
    for ep in range(1, args.epochs+1):
        run_epoch(enc_h, head_h, True,  dl, crit_h, opt_h, dev)
        run_epoch(enc_s, head_s, False, dl, crit_s, opt_s, dev)
        if ep % 50 == 0:
            _, a_h = run_epoch(enc_h, head_h, True,  dl, crit_h, None, dev)
            _, a_s = run_epoch(enc_s, head_s, False, dl, crit_s, None, dev)
            logging.info("Ep %3d  train‐acc HarMax=%.3f  Softmax=%.3f", ep, a_h, a_s)

    # ── embed & score test ─────────────────────────────────
    enc_h.eval(); head_h.eval()
    enc_s.eval(); head_s.eval()
    with torch.no_grad():
        H = enc_h(torch.tensor(X_test).to(dev)).cpu()
        P_h = head_h(H).cpu().numpy()
        S = enc_s(torch.tensor(X_test).to(dev)).cpu()
        logits = head_s(S).cpu().numpy()

    score_h = 1.0 - P_h.max(axis=1)
    proba_s = np.exp(logits) / np.exp(logits).sum(axis=1,keepdims=True)
    score_s = 1.0 - proba_s.max(axis=1)

    # ── compute metrics ────────────────────────────────────
    auc_h = roc_auc_score(y_test_bin, score_h)
    ap_h  = average_precision_score(y_test_bin, score_h)
    auc_s = roc_auc_score(y_test_bin, score_s)
    ap_s  = average_precision_score(y_test_bin, score_s)

    print(f"\nResults on `{args.dataset}`, held‐out={novel}")
    print(f" HarMax AUC={auc_h:.3f}   AP={ap_h:.3f}")
    print(f" Softmax AUC={auc_s:.3f}   AP={ap_s:.3f}")

    # ── save metrics ───────────────────────────────────────
    out_res = pathlib.Path("results")/"anomaly"
    out_res.mkdir(parents=True, exist_ok=True)
    res = {
        "dataset":   args.dataset,
        "held_out":  novel,
        "n_known":   K,
        "roc_auc":   {"harmax":auc_h, "softmax":auc_s},
        "avg_prec":  {"harmax":ap_h,  "softmax":ap_s}
    }
    with (out_res/f"{args.dataset}_held{novel}.json").open("w") as f:
        json.dump(res, f, indent=2)

    # ── plot ROC ──────────────────────────────────────────
    from sklearn.metrics import roc_curve
    fpr_h, tpr_h, _ = roc_curve(y_test_bin, score_h)
    fpr_s, tpr_s, _ = roc_curve(y_test_bin, score_s)
    fig, ax = plt.subplots()
    ax.plot(fpr_h, tpr_h, label=f"HarMax (AUC={auc_h:.3f})")
    ax.plot(fpr_s, tpr_s, label=f"Softmax (AUC={auc_s:.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{args.dataset} anomaly ROC")
    ax.legend(loc="lower right")

    out_fig = pathlib.Path("figures")/"anomaly"
    out_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig/f"{args.dataset}_held{novel}_roc.png", dpi=120)
    plt.close(fig)


if __name__=="__main__":
    main()
