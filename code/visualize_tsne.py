"""t-SNE visualization of the CNN penultimate-layer representation.

Uses the post-ReLU 128-d activation from a frozen fold model (default
seed-42 fold-1) for all 320 samples, then projects to 2-D with t-SNE.
Descriptive only: the displayed fold includes training samples, so the
clustering partly reflects seen data and is NOT independent evidence of
generalization.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold

from model import CNN1D

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(os.path.join(args.data_dir, "X.npy"))
    y = np.load(os.path.join(args.data_dir, "y.npy"))

    model = CNN1D(input_length=500, num_classes=4).to(device)
    ckpt = torch.load(os.path.join(args.model_dir, "best_model.pth"),
                       map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # post-ReLU penultimate activation (fc1 + ReLU, 128-d, non-negative)
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            xb = torch.tensor(X[i:i+64], dtype=torch.float32,
                              device=device).unsqueeze(1)
            x = torch.relu(model.conv1(xb))
            x = torch.relu(model.conv2(x))
            x = model.pool(x)
            x = model.dropout(x)
            x = x.view(x.size(0), -1)
            emb = torch.relu(model.fc1(x))
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)

    tsne = TSNE(n_components=2, perplexity=args.perplexity,
                random_state=args.seed, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(embeddings)

    # fold-1 split (StratifiedKFold, same seed) to mark training vs held-out
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(iter(skf.split(X, y)))
    is_train = np.zeros(len(y), dtype=bool)
    is_train[train_idx] = True
    n_train, n_heldout = int(is_train.sum()), len(y) - int(is_train.sum())

    coord_df = pd.DataFrame({
        "array_index": range(len(y)),
        "label": [CLASS_NAMES[c] for c in y],
        "split": ["train" if is_train[i] else "held-out" for i in range(len(y))],
        "tsne_1": emb_2d[:, 0], "tsne_2": emb_2d[:, 1]})
    coord_df.to_csv(args.out_csv, index=False)
    np.save(args.out_npy, embeddings)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for ci, cname in enumerate(CLASS_NAMES):
        m = y == ci
        ax.scatter(emb_2d[m & is_train, 0], emb_2d[m & is_train, 1],
                   c=COLORS[ci], marker="o", s=22, alpha=0.55, label=cname)
        ax.scatter(emb_2d[m & ~is_train, 0], emb_2d[m & ~is_train, 1],
                   c=COLORS[ci], marker="^", s=75, alpha=0.95,
                   edgecolors="k", linewidths=0.4)
    train_h = mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                            markersize=7, alpha=0.6, label=f"training ({n_train})")
    ho_h = mlines.Line2D([], [], color="gray", marker="^", linestyle="None",
                         markersize=9, markeredgecolor="k",
                         label=f"held-out ({n_heldout})")
    leg_class = ax.legend(title="gas class", loc="upper left")
    ax.add_artist(leg_class)
    ax.legend(handles=[train_h, ho_h], title="fold-1 split", loc="lower right")
    ax.set_title("t-SNE of CNN penultimate-layer representation", fontsize=11)
    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    plt.close(fig)
    print(f"t-SNE done: {embeddings.shape} -> 2D; "
          f"{n_train} training / {n_heldout} held-out")
    print(f"  -> {args.out_png}, {args.out_csv}, {args.out_npy}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="t-SNE of CNN penultimate representation.")
    p.add_argument("--model_dir", type=str,
                   default="results/cnn_seed42/fold1")
    p.add_argument("--data_dir", type=str, default="data/preprocessed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=int, default=30)
    p.add_argument("--out_png", type=str, default="tsne_cnn_embedding.png")
    p.add_argument("--out_csv", type=str, default="tsne_coordinates.csv")
    p.add_argument("--out_npy", type=str, default="tsne_embedding_128d.npy")
    args = p.parse_args()
    main(args)
