# utils_plotting.py

import pathlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import zscore
from typing import Sequence
import torch

def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            h = model.encoder(X)
            embeddings.append(h.cpu())
            labels.append(y)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return embeddings, labels


def plot_embeddings_comparison(
    harmax_embeddings: np.ndarray,
    softmax_embeddings: np.ndarray,
    labels: np.ndarray,
    harmax_centres: np.ndarray,
    softmax_weights: np.ndarray,
    save_path: pathlib.Path,
    figsize=(10,10),
    tsne_perplexity=30,
    random_state=0
):
    """
    Produces a 2×2 figure:
      (a) Harmax PCA
      (b) Harmax t-SNE
      (c) Softmax PCA
      (d) Softmax t-SNE
    and saves to `save_path`.
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # ─── Harmax PCA ────────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    har_pca = pca.fit_transform(harmax_embeddings)
    ev_har = pca.explained_variance_ratio_.sum() * 100
    centers_har_pca = pca.transform(harmax_centres)
    ax = axs[0,0]
    for cls in np.unique(labels):
        m = labels == cls
        ax.scatter(har_pca[m,0], har_pca[m,1], s=30, alpha=0.6, label=f'Class {cls}')
    ax.scatter(centers_har_pca[:,0], centers_har_pca[:,1],
               c='k', marker='*', s=200, label='Centers')
    ax.set_title(f'(a) Harmax PCA – EV {ev_har:.1f}%')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(); ax.set_aspect('equal')

    # ─── Harmax t-SNE ─────────────────────────────────────────────────────
    combo = np.vstack([harmax_embeddings, harmax_centres])
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state)
    combo_tsne = tsne.fit_transform(combo)
    data_tsne, centers_tsne = combo_tsne[:len(harmax_embeddings)], combo_tsne[len(harmax_embeddings):]
    ax = axs[0,1]
    for cls in np.unique(labels):
        m = labels == cls
        ax.scatter(data_tsne[m,0], data_tsne[m,1], s=30, alpha=0.6, label=f'Class {cls}')
    ax.scatter(centers_tsne[:,0], centers_tsne[:,1],
               c='k', marker='*', s=200, label='Centers')
    ax.set_title('(b) Harmax t-SNE'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(); ax.set_aspect('equal')

    # ─── Softmax PCA ───────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    soft_pca = pca.fit_transform(softmax_embeddings)
    ev_soft = pca.explained_variance_ratio_.sum() * 100
    centers_soft_pca = pca.transform(softmax_weights)
    ax = axs[1,0]
    for cls in np.unique(labels):
        m = labels == cls
        ax.scatter(soft_pca[m,0], soft_pca[m,1], s=30, alpha=0.6, label=f'Class {cls}')
    ax.scatter(centers_soft_pca[:,0], centers_soft_pca[:,1],
               c='k', marker='*', s=200, label='Weights')
    ax.set_title(f'(c) Softmax PCA – EV {ev_soft:.1f}%')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(); ax.set_aspect('equal')

    # ─── Softmax t-SNE ────────────────────────────────────────────────────
    combo = np.vstack([softmax_embeddings, softmax_weights])
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state)
    combo_tsne = tsne.fit_transform(combo)
    data_tsne, centers_tsne = combo_tsne[:len(softmax_embeddings)], combo_tsne[len(softmax_embeddings):]
    ax = axs[1,1]
    for cls in np.unique(labels):
        m = labels == cls
        ax.scatter(data_tsne[m,0], data_tsne[m,1], s=30, alpha=0.6, label=f'Class {cls}')
    ax.scatter(centers_tsne[:,0], centers_tsne[:,1],
               c='k', marker='*', s=200, label='Weights')
    ax.set_title('(d) Softmax t-SNE'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(); ax.set_aspect('equal')

    plt.tight_layout()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_prototypes(
    X_raw: np.ndarray,
    labels: np.ndarray,
    harmax_embeddings: np.ndarray,
    harmax_centres: np.ndarray,
    softmax_embeddings: np.ndarray,
    softmax_weights: np.ndarray,
    save_path: pathlib.Path,
    figsize=(10,5)
):
    """
    Finds the nearest in-class prototype for each centre/weight,
    then plots 2×2 (top = Harmax, bottom = Softmax).
    """
    def _closest(emb, centres):
        idxs = []
        for j, c in enumerate(centres):
            cls_idx = np.where(labels == j)[0]
            d = np.linalg.norm(emb[cls_idx] - c, axis=1)
            idxs.append(cls_idx[np.argmin(d)])
        return idxs

    h_idx = _closest(harmax_embeddings, harmax_centres)
    s_idx = _closest(softmax_embeddings, softmax_weights)

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    for row, (idxs, title) in enumerate(zip([h_idx, s_idx], ['Harmax', 'Softmax'])):
        for i, idx in enumerate(idxs):
            ax = axs[row, i]
            cls = i
            for j, x in enumerate(X_raw):
                if labels[j] == cls:
                    ax.plot(x, color='gray', alpha=0.1)
            ax.plot(X_raw[idx], color='blue', lw=2,
                    label=f'Prototype idx={idx}')
            ax.set_title(f'{title} Class {cls}')
            ax.set_xlabel('Time step'); ax.set_ylabel('Value')
            if row==0 and i==0: ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

