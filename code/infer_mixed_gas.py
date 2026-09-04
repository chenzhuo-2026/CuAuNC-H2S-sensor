"""Frozen-CNN ensemble inference on mixed-gas recordings (no retraining).

Uses the trained four-class CNN (H2S/CH4/CO2/AIR), frozen seed-42 five-fold
models (results/cnn_seed42). Per sample it exports: ensemble softmax
(five-fold mean, 4-d), predicted class (argmax), per-fold raw logits (4-d),
and per-fold penultimate embedding (128-d, fc1 output, for t-SNE). Reports
per-class softmax confidence mean +/- SD with bar plots.
"""
import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CNN1D

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]
# mixture type -> human-readable label (reporting only, not used in training)
MIX_LABEL_MAP = {
    "MIX_H2S_CO2":     "H2S+CO2",
    "MIX_H2S_CH4":     "H2S+CH4",
    "MIX_H2S_CH4_CO2": "H2S+CH4+CO2",
    "MIX_H2S_SO2":     "H2S+SO2",
    "MIX_H2S_SO2_CO2": "H2S+SO2+CO2",
}


def load_fold_models(model_dir, input_length, device):
    models = []
    for k in range(1, 6):
        m = CNN1D(input_length=input_length, num_classes=len(CLASS_NAMES)).to(device)
        ckpt = torch.load(os.path.join(model_dir, f"fold{k}", "best_model.pth"),
                          map_location=device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
    return models


def forward_get_internals(model, x, embedding_hook):
    """Single forward pass, returns (logits[4], embedding[128]) via fc1 hook."""
    handles = []
    captured = {}

    def hook(mod, inp, out):
        captured["emb"] = out.detach()

    # fc1 output (pre-ReLU, 128-d)
    if embedding_hook:
        handles.append(model.fc1.register_forward_hook(hook))
    with torch.no_grad():
        logits = model(x)
    for h in handles:
        h.remove()
    emb = captured.get("emb", None)
    return logits, emb


@torch.no_grad()
def infer_one_mixture(models, X_np, device, batch_size=64):
    """Run five folds for one mixture's X (N,500); returns internal tensors."""
    X = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1,500)
    N = X.shape[0]
    all_logits = []      # (F, N, 4)
    all_emb = []          # (F, N, 128)
    for m in models:
        logits_list, emb_list = [], []
        for i in range(0, N, batch_size):
            xb = X[i:i+batch_size]
            logits, emb = forward_get_internals(m, xb, embedding_hook=True)
            logits_list.append(logits.cpu().numpy())
            emb_list.append(emb.cpu().numpy())
        all_logits.append(np.concatenate(logits_list, axis=0))
        all_emb.append(np.concatenate(emb_list, axis=0))
    logits = np.stack(all_logits, axis=0)   # (F, N, 4)
    emb = np.stack(all_emb, axis=0)         # (F, N, 128)
    # ensemble: five-fold softmax mean
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()  # (F, N, 4)
    ens_probs = probs.mean(axis=0)          # (N, 4)
    preds = ens_probs.argmax(axis=1)        # (N,)
    return logits, emb, ens_probs, preds


def confidence_stats(ens_probs, preds):
    """Per-class confidence mean +/- SD.

    Groups by predicted class and reports that class's softmax confidence
    mean +/- SD; also reports each fixed class's probability mean +/- SD
    across all samples. Returns two DataFrames.
    """
    rows_group = []
    for c in range(len(CLASS_NAMES)):
        mask = preds == c
        if mask.sum() == 0:
            rows_group.append({"class": CLASS_NAMES[c], "n": 0,
                               "conf_mean": np.nan, "conf_sd": np.nan})
        else:
            conf = ens_probs[mask, c]
            rows_group.append({"class": CLASS_NAMES[c], "n": int(mask.sum()),
                               "conf_mean": float(conf.mean()),
                               "conf_sd": float(conf.std(ddof=0))})
    df_group = pd.DataFrame(rows_group)

    # each fixed class probability mean+/-SD across all samples
    rows_col = []
    for c in range(len(CLASS_NAMES)):
        p = ens_probs[:, c]
        rows_col.append({"class": CLASS_NAMES[c],
                         "prob_mean": float(p.mean()),
                         "prob_sd": float(p.std(ddof=0))})
    df_col = pd.DataFrame(rows_col)
    return df_group, df_col


def plot_bar(df, title, fname, value_col, sd_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    means = np.nan_to_num(df[value_col].values.astype(float), nan=0.0)
    sds = np.nan_to_num(df[sd_col].values.astype(float), nan=0.0)
    ax.bar(df["class"], means, yerr=sds, capsize=5,
           color=["#d62728", "#1f77b4", "#2ca02c", "#7f7f7f"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Softmax probability")
    ax.set_title(title)
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + 0.02, f"{m:.2f}±{s:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    models = load_fold_models(args.model_dir, input_length=500, device=device)

    os.makedirs(args.output_dir, exist_ok=True)

    mix_types = args.mix_types or sorted(
        f[2:-4] for f in os.listdir(args.data_dir)
        if f.startswith("X_MIX") and f.endswith(".npy"))
    print(f"mixtures to infer: {mix_types}\n")

    summary_rows = []
    for mt in mix_types:
        X = np.load(os.path.join(args.data_dir, f"X_{mt}.npy"))
        print(f"==== {mt}  X.shape={X.shape} ====")
        logits, emb, ens_probs, preds = infer_one_mixture(models, X, device)

        out = os.path.join(args.output_dir, mt)
        os.makedirs(out, exist_ok=True)
        # save per-sample raw logits (5 folds) + embedding (5 folds) + ensemble softmax + preds
        np.save(os.path.join(out, "logits_all_folds.npy"), logits)        # (5,N,4)
        np.save(os.path.join(out, "embedding_all_folds.npy"), emb)        # (5,N,128)
        np.save(os.path.join(out, "softmax_ensemble.npy"), ens_probs)     # (N,4)
        np.save(os.path.join(out, "preds.npy"), preds)                    # (N,)

        # per-sample detail CSV
        df = pd.DataFrame(ens_probs, columns=[f"p_{c}" for c in CLASS_NAMES])
        df.insert(0, "pred_class", [CLASS_NAMES[p] for p in preds])
        df.insert(0, "sample_idx", np.arange(len(preds)))
        df.to_csv(os.path.join(out, "per_sample_softmax.csv"), index=False)

        df_group, df_col = confidence_stats(ens_probs, preds)
        df_group.to_csv(os.path.join(out, "confidence_by_pred_class.csv"), index=False)
        df_col.to_csv(os.path.join(out, "prob_by_class.csv"), index=False)

        plot_bar(df_group, f"{MIX_LABEL_MAP.get(mt, mt)}: confidence by predicted class",
                 os.path.join(out, "bar_confidence_by_pred_class.png"),
                 "conf_mean", "conf_sd")
        plot_bar(df_col, f"{MIX_LABEL_MAP.get(mt, mt)}: mean probability per class",
                 os.path.join(out, "bar_prob_per_class.png"),
                 "prob_mean", "prob_sd")

        print(df_group.to_string(index=False))
        print(f"pred distribution: {dict(zip(CLASS_NAMES, [int((preds==c).sum()) for c in range(4)]))}")
        print(f"-> saved to {out}\n")

        for _, r in df_col.iterrows():
            summary_rows.append({"mixture": MIX_LABEL_MAP.get(mt, mt),
                                 "class": r["class"],
                                 "prob_mean": r["prob_mean"],
                                 "prob_sd": r["prob_sd"],
                                 "n_samples": len(preds)})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.output_dir, "summary_all_mixtures.csv"), index=False)
    print("=" * 70)
    print("Summary (per-class probability mean +/- SD, by mixture):")
    print(summary.to_string(index=False))
    print(f"\nAll outputs under {args.output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="results/cnn_seed42")
    p.add_argument("--data_dir", default="data/preprocessed_mixed")
    p.add_argument("--output_dir", default="results/mixed_gas")
    p.add_argument("--mix_types", nargs="+", default=None)
    args = p.parse_args()
    main(args)
