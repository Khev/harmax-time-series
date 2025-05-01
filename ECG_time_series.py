#!/usr/bin/env python3
# ecg200_harmax_vs_softmax.py
# ---------------------------------------------------------------------
# Compare soft-max vs. harmax on the ECG200 data set
# ---------------------------------------------------------------------
import argparse, math, pathlib, random, sys, json
from typing import Tuple

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore

# ------------------------------- utils --------------------------------
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_ucr_txt(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a UCR .txt file irrespective of whether it is comma- or
    whitespace-separated.  Returns:
        X  – (N, L) float32
        y  – (N,)   int64   (converts labels to 0 … C-1 internally)
    """
    series, labels = [], []
    with open(path, "r") as f:
        for ln in f:
            # split on comma if any, else on whitespace
            parts = ln.replace('\t', ' ').strip().split(',')
            if len(parts) == 1:                    # comma not found
                parts = [p for p in ln.split() if p]

            label_str, *values = parts
            labels.append(int(float(label_str)))   # ECG200 labels are -1 / 1
            series.append([float(v) for v in values])

    X = np.asarray(series, dtype=np.float32)
    y_raw = np.asarray(labels, dtype=np.int64)

    # map arbitrary integer labels → consecutive {0,1,…,C-1}
    uniq = {lab: i for i, lab in enumerate(sorted(set(y_raw)))}
    y = np.vectorize(uniq.get)(y_raw).astype(np.int64)
    return X, y


# ----------------------------- model bits -----------------------------
class Encoder1D(nn.Module):
    """Very small 1-D CNN encoder (similar to FCN-Baseline)."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),          # (B,128)
            nn.Linear(128, out_dim), nn.ReLU()
        )

    def forward(self, x):          # x: (B, L)
        return self.net(x.unsqueeze(1))  # (B, out_dim)

class SoftmaxHead(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.logits = nn.Linear(in_dim, n_cls)

    def forward(self, h):
        return self.logits(h)

class HarmaxHead(nn.Module):
    """Learnable class centres  C∈ ℝ^{nCls×d}; p_i ∝ 1/‖h-C_i‖^n"""
    def __init__(self, in_dim, n_cls, n_exp=None):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(n_cls, in_dim)/math.sqrt(in_dim))
        self.n_exp   = n_exp or int(math.sqrt(in_dim))

    def probs(self, h):
        d = torch.cdist(h, self.centres) + 1e-8      # (B,n_cls)
        inv = d.pow(-self.n_exp)
        return inv / inv.sum(dim=1, keepdim=True)

    def forward(self, h):    # returns probs not logits
        return self.probs(h)

class HarmonicLoss(nn.Module):
    def forward(self, probs, target):
        return -torch.log(probs[torch.arange(probs.size(0)), target]).mean()

# ----------------------------- training --------------------------------
def run_epoch(model, is_harmax, loader, criterion, optim, device):
    train = optim is not None
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if train: optim.zero_grad()
        h = model.encoder(X)
        if is_harmax:
            probs = model.head(h)
            loss  = criterion(probs, y)
            pred  = probs.argmax(1)
        else:
            logits = model.head(h)
            loss   = criterion(logits, y)
            pred   = logits.argmax(1)
        if train:
            loss.backward()
            optim.step()
        loss_sum += loss.item() * y.size(0)
        correct  += (pred == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total

class Wrapper(nn.Module):
    def __init__(self, encoder, head, is_harmax=False):  # convenience container
        super().__init__()
        self.encoder, self.head, self.is_harmax = encoder, head, is_harmax

# ----------------------------- main script ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--epochs',  type=int, default=400)
    p.add_argument('--batch',   type=int, default=32)
    p.add_argument('--lr',      type=float, default=3e-3)
    p.add_argument('--wd',      type=float, default=1e-2)
    p.add_argument('--device',  type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',    type=int, default=0)
    p.add_argument('--n_exp', type=float, default=None,
                help='harmonic exponent n (default √d)')
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    data_dir = pathlib.Path(args.data_dir)
    Xtr, ytr = load_ucr_txt(data_dir/'ECG200_TRAIN.txt')
    Xte, yte = load_ucr_txt(data_dir/'ECG200_TEST.txt')
    Xtr, Xte = zscore(Xtr), zscore(Xte)
    #Xtr, Xte = zscore(Xtr, axis=1), zscore(Xte, axis=1)

    n_cls, length = len(np.unique(ytr)), Xtr.shape[1]
    print(f"Loaded ECG200: train {Xtr.shape}  test {Xte.shape}  classes={n_cls}")

    tr_dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                       batch_size=args.batch, shuffle=True, drop_last=True)
    te_dl = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
                       batch_size=args.batch)

    encoder = Encoder1D(out_dim=64)

    models = {
        'softmax': Wrapper(encoder=Encoder1D(64), head=SoftmaxHead(64, n_cls), is_harmax=False),
        'harmax' : Wrapper(Encoder1D(64), HarmaxHead(64, n_cls, args.n_exp), is_harmax=True)
    }
    criteria = {
        'softmax': nn.CrossEntropyLoss(),
        'harmax' : HarmonicLoss()
    }
    opts = {
        name: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=(args.wd if name=='softmax' else 0.0))
        for name,m in models.items()
    }
    hist = {name:{'tr':[],'te':[]} for name in models}
    # Training loop -----------------------------------------------------
    for epoch in range(1, args.epochs+1):
        for name, model in models.items():
            model.to(device)
            tr_loss, tr_acc = run_epoch(model, model.is_harmax, tr_dl, criteria[name], opts[name], device)
            te_loss, te_acc = run_epoch(model, model.is_harmax, te_dl, criteria[name], None, device)
            hist[name]['tr'].append(tr_acc);  hist[name]['te'].append(te_acc)
        if epoch%50==0 or epoch==1:
            s = " | ".join([f"{n}: {hist[n]['te'][-1]*100:.1f}%" for n in models])
            print(f"Epoch {epoch:3d}  test-acc  {s}")

    # -------------------------------- plots ---------------------------
    xs = np.arange(1, args.epochs + 1)

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # --- 1 ▸ soft-max panel ------------------------------------------ #
    axs[0].plot(xs, hist['softmax']['tr'],  color='tab:blue',   label='train')
    axs[0].plot(xs, hist['softmax']['te'],  color='tab:orange', label='test')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('softmax')
    axs[0].set_ylim(0, 1.05)
    axs[0].legend(loc='lower right', fontsize=8)

    # --- 2 ▸ harmax panel -------------------------------------------- #
    axs[1].plot(xs, hist['harmax']['tr'],   color='tab:blue',   label='train')
    axs[1].plot(xs, hist['harmax']['te'],   color='tab:orange', label='test')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title(f'harmax (n={args.n_exp if args.n_exp is not None else int(math.sqrt(64))})')
    axs[1].set_ylim(0, 1.05)
    axs[1].legend(loc='lower right', fontsize=8)

    axs[1].set_xlabel('Epoch')
    fig.suptitle('ECG200 classification', y=0.95)
    fig.tight_layout()

    fig_path = pathlib.Path('figures') / f'ecg200_acc_{args.n_exp}.png'
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=120)
    print(f"Saved {fig_path}")


    # ---------- optional PCA of class-vectors ------------------------
    def class_vecs(mod):
        if mod.is_harmax: return mod.head.centres.detach().cpu()
        else:             return mod.head.logits.weight.detach().cpu()
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    for i,(name,mod) in enumerate(models.items()):
        vec = class_vecs(mod)
        pcs = PCA(2).fit_transform(vec)
        ev  = PCA(2).fit(vec).explained_variance_ratio_.sum()*100
        ax[i].scatter(pcs[:,0], pcs[:,1], s=30)
        for j,(x,y) in enumerate(pcs): ax[i].text(x,y,str(j),ha='center',va='center',fontsize=7)
        ax[i].set_aspect('equal'); ax[i].set_title(f"{name}  EV:{ev:.0f}%")
    plt.tight_layout(); plt.savefig(f'figures/ecg200_pca_{args.n_exp}.png', dpi=120)
    print('Saved ecg200_pca.png')

    # --- emit history for the tuning wrapper ---------------------------
    print("RESULTS:", json.dumps({
            "harmax_tr": hist['harmax']['tr'],
            "harmax_te": hist['harmax']['te'],
            "soft_tr"  : hist['softmax']['tr'],
            "soft_te"  : hist['softmax']['te']
    }))

if __name__ == "__main__":
    main()

