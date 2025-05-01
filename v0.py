#!/usr/bin/env python3
"""
modular_addition_experiment.py

Recreates the 'adding integers modulo' experiment from
"Harmonic Loss Trains Interpretable AI Models" (Baek et al., 2025).

• Dataset:  (x, y) ↦ (x + y) mod 31,  x, y ∈ {0,…,30}
• Models :  (1) Standard MLP + Cross-Entropy
            (2) Harmonic-MLP   + Harmonic loss (n = 1)
Outputs:    train / test accuracy per epoch + learning-curve figure.

Run:
    python modular_addition_experiment.py
Optional CLI flags (see --help):
    --epochs, --batch, --lr, --wd, --device
"""

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------- Hyper-parameters ----------------------------- #
N_CLASSES = 31      # modulus
EMBED_DIM = 16      # dim used in Baek et al. for MLP experiments
HIDDEN1   = 100
HIDDEN2   = 16
HARMONIC_N = 1      # exponent n in paper (they used n=1 for MLP section)
SEED = 42
FIG_DIR = Path("figures")
# --------------------------------------------------------------------------- #


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------- Dataset construction --------------------------- #
def build_dataset(train_frac: float = 0.8) -> Tuple[TensorDataset, TensorDataset]:
    """Construct full cartesian product of (x, y) ∈ {0,…,30}² with labels."""
    xs, ys, labels = [], [], []
    for x in range(N_CLASSES):
        for y in range(N_CLASSES):
            xs.append(x)
            ys.append(y)
            labels.append((x + y) % N_CLASSES)
    X = torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    # Shuffle once, then split
    indices = torch.randperm(len(X))
    split = int(train_frac * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    test_ds  = TensorDataset(X[test_idx],  y[test_idx])
    return train_ds, test_ds


# ------------------------------ Model blocks -------------------------------- #
class Embed2D(nn.Module):
    """
    Turns the two integer inputs (x,y) into a single vector by
    summing their embeddings (shared table of size N_CLASSES × EMBED_DIM).
    """
    def __init__(self, vocab=N_CLASSES, dim=EMBED_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, xy: torch.LongTensor) -> torch.Tensor:
        x_embed = self.emb(xy[:, 0])
        y_embed = self.emb(xy[:, 1])
        return x_embed + y_embed


class StandardMLP(nn.Module):
    """MLP with CE logits."""
    def __init__(self):
        super().__init__()
        self.encoder = Embed2D()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN1),
            nn.SiLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.SiLU(),
        )
        self.out = nn.Linear(HIDDEN2, N_CLASSES)  # CE logits

    def forward(self, xy):
        h = self.net(self.encoder(xy))
        logits = self.out(h)              # (B, 31)
        return logits


class HarmonicMLP(nn.Module):
    """
    MLP that leaves the penultimate representation h ∈ ℝ^{16},
    and learns class-centers C ∈ ℝ^{31×16}.  Probabilities:

        p_i = (1 / ‖h − C_i‖^n) / Σ_j …

    Here n = 1 (paper’s MLP setting).
    """
    def __init__(self, n_exp=HARMONIC_N):
        super().__init__()
        self.n_exp = n_exp
        self.encoder = Embed2D()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN1),
            nn.SiLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.SiLU(),
        )
        # Class centres, initialised from N(0,1/√d)
        self.centres = nn.Parameter(torch.randn(N_CLASSES, HIDDEN2) / math.sqrt(HIDDEN2))

    def forward(self, xy):
        h = self.net(self.encoder(xy))            # (B, 16)
        return h                                  # caller handles loss / pred

    # --------- Convenience helpers ----------------------------------------- #
    def harmonic_probs(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, d), centres: (31, d) → distances (B,31)
        dists = torch.cdist(h, self.centres, p=2) + 1e-8
        inv_pow = dists.pow(-self.n_exp)          # 1 / d^n
        probs = inv_pow / inv_pow.sum(dim=1, keepdim=True)
        return probs

    def predict(self, xy):
        h = self.forward(xy)
        probs = self.harmonic_probs(h)
        return probs.argmax(dim=1)


# ---------------------------- Training helpers ------------------------------ #
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if isinstance(model, HarmonicMLP):
            h = model(x)
            probs = model.harmonic_probs(h)
            loss = criterion(probs, y)
            pred = probs.argmax(dim=1)
        else:
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct  += (pred == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, HarmonicMLP):
            h = model(x)
            probs = model.harmonic_probs(h)
            loss = criterion(probs, y)
            pred = probs.argmax(dim=1)
        else:
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)

        loss_sum += loss.item() * y.size(0)
        correct  += (pred == y).sum().item()
        total    += y.size(0)
    return loss_sum / total, correct / total


# --------------------------- Loss abstractions ------------------------------ #
class HarmonicLoss(nn.Module):
    def forward(self, probs: torch.Tensor, target: torch.Tensor):
        # probs: (B,31), target: (B,)
        return -torch.log(probs[torch.arange(probs.size(0)), target] + 1e-12).mean()


# ---------------------------------- Main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Modular-Addition Harmonic vs. CE")
    parser.add_argument("--epochs", type=int, default=7000,
                        help="Epochs (Baek et al. used 7000 for MLPs)")
    parser.add_argument("--batch",  type=int, default=256)
    parser.add_argument("--lr",     type=float, default=2e-3)
    parser.add_argument("--wd",     type=float, default=1e-2,
                        help="Weight-decay (Baek et al. used 1e-2)")
    parser.add_argument("--device", type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(args.device)

    train_ds, test_ds = build_dataset()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch)

    # Prepare models, criteria, optimizers
    models = {
        "Standard":  StandardMLP().to(device),
        "Harmonic":  HarmonicMLP().to(device)
    }
    criteria = {
        "Standard":  nn.CrossEntropyLoss(),
        "Harmonic":  HarmonicLoss()
    }
    optims = {
        name: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
        for name, m in models.items()
    }

    history = {name: {"train_acc": [], "test_acc": []} for name in models}

    for epoch in range(1, args.epochs + 1):
        for name, model in models.items():
            train_loss, train_acc = train_epoch(model, train_loader,
                                                criteria[name], optims[name], device)
            test_loss,  test_acc  = evaluate(model, test_loader,
                                             criteria[name], device)
            history[name]["train_acc"].append(train_acc)
            history[name]["test_acc"].append(test_acc)

        # Show progress every 100 epochs
        if epoch % 100 == 0 or epoch == 1 or epoch == args.epochs:
            msg = (f"Epoch {epoch:>4}/{args.epochs}:  "
                   f"Std acc  {history['Standard']['test_acc'][-1]*100:5.1f}%  |  "
                   f"Harm acc {history['Harmonic']['test_acc'][-1]*100:5.1f}%")
            print(msg)

    # ---------------------------- Plot learning curve ----------------------- #
    FIG_DIR.mkdir(exist_ok=True)
    xs = np.arange(1, args.epochs + 1)

    plt.figure(figsize=(6,4))
    for name, style in zip(history, ["--", "-"]):
        plt.plot(xs, history[name]["test_acc"], linestyle=style, label=f"{name} test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Modular-Addition: Test Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    fig_path = FIG_DIR / "modular_addition_learning_curve.png"
    plt.savefig(fig_path, dpi=120)
    print(f"\nLearning-curve saved to {fig_path.absolute()}")


if __name__ == "__main__":
    main()

