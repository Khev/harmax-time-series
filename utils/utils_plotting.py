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
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(loc='best'); ax.set_aspect('equal')

    # ─── Harmax t-SNE ─────────────────────────────────────────────────────
    combo = np.vstack([harmax_embeddings, harmax_centres])
    n_pts = combo.shape[0]
    perp = min(tsne_perplexity, max(5, (n_pts - 1)//3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
    combo_tsne = tsne.fit_transform(combo)
    data_tsne, centers_tsne = combo_tsne[:len(harmax_embeddings)], combo_tsne[len(harmax_embeddings):]
    ax = axs[0,1]
    for cls in np.unique(labels):
        m = labels == cls
        ax.scatter(data_tsne[m,0], data_tsne[m,1], s=30, alpha=0.6, label=f'Class {cls}')
    ax.scatter(centers_tsne[:,0], centers_tsne[:,1],
               c='k', marker='*', s=200, label='Centers')
    ax.set_title('(b) Harmax t-SNE'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best'); ax.set_aspect('equal')

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
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(loc='best'); ax.set_aspect('equal')

    # ─── Softmax t-SNE ────────────────────────────────────────────────────
    combo = np.vstack([softmax_embeddings, softmax_weights])
    # reuse the same perp and random_state
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
    combo_tsne = tsne.fit_transform(combo)
    data_tsne, centers_tsne = combo_tsne[:len(softmax_embeddings)], combo_tsne[len(softmax_embeddings):]
    ax = axs[1,1]
    for cls in np.unique(labels):
        m = labels == cls
        ax.scatter(data_tsne[m,0], data_tsne[m,1], s=30, alpha=0.6, label=f'Class {cls}')
    ax.scatter(centers_tsne[:,0], centers_tsne[:,1],
               c='k', marker='*', s=200, label='Weights')
    ax.set_title('(d) Softmax t-SNE'); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best'); ax.set_aspect('equal')

    plt.tight_layout()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)



def plot_embeddings_comparison_old(
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
    perp = max(5, min(30, (n_pts-1)//3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=0)
    #tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state)
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
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
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
    figsize_unit=(5,5)
):
    """
    Finds the nearest in-class prototype for each centre/weight,
    then plots 2 rows (harmax, softmax) × n_classes columns.
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
    n_cls = len(h_idx)

    fig, axs = plt.subplots(2, n_cls, figsize=(figsize_unit[0]*n_cls, figsize_unit[1]*2))
    for row, (idxs, title) in enumerate(zip([h_idx, s_idx], ['Harmax', 'Softmax'])):
        for i, idx in enumerate(idxs):
            ax = axs[row, i] if n_cls > 1 else axs[row]
            cls = i
            # plot all series of this class
            for j, x in enumerate(X_raw):
                if labels[j] == cls:
                    ax.plot(x, color='gray', alpha=0.1)
            # overlay prototype
            ax.plot(X_raw[idx], color='blue', lw=2)
            ax.set_title(f"{title} Class {cls} (idx={idx})")
            ax.set_xlabel('Timestep'); ax.set_ylabel('Value')
    plt.tight_layout()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

def plot_prototypes_old(
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



def load_dataset_aeon(name, split="train"):
    if load_classification is None:
        raise RuntimeError("aeon missing – install it or use --data_dir")
    try:
        X, y = load_classification(name, split=split, return_type="numpy3d")
    except TypeError:
        X, y = load_classification(name, split=split)
    if isinstance(X, np.ndarray) and X.ndim == 3:
        X = X[:, 0, :]
    else:
        from aeon.utils import convert_series_to_array
        X = convert_series_to_array(X)
    return X.astype("float32"), y.astype("int64")

