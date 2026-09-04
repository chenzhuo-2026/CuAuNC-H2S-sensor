"""Frozen-CNN ensemble inference on a single raw CSV recording (no retraining).

Reuses the training preprocessing: K-means anomaly removal -> per-window
Min-Max normalization -> air-baseline linear detrending -> interpolation to
500 points -> Gaussian smoothing (sigma=1.0). Models are the frozen seed-42
five-fold CNN (results/cnn_seed42/fold{1-5}/best_model.pth).

Usage:
  python infer_unseen_h2s.py --csv "data/raw_data/unseen_H2S/20ppb.csv" --gas H2S
"""
import os
import argparse
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn.functional as F

from model import CNN1D

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]
# per-gas preprocessing config: (n_clusters, edge_sec, all_air)
GAS_CFG = {
    "H2S": (4, 0.2, False), "CH4": (2, 0.15, False), "CO2": (2, 0.25, False),
    "AIR": (2, 1.0, True), "MIX": (4, 0.2, False),
}


def read_csv(path):
    df = pd.read_csv(path, skiprows=3, header=None, names=["TIME(S)", "R(OHM)"])
    for c in ["TIME(S)", "R(OHM)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def remove_anomaly(df, n_clusters, edge_sec):
    """K-means-based anomaly removal (replicates data_process logic, no plotting)."""
    df = df.copy()
    df['diff'] = df['R(OHM)'].diff().abs().fillna(0)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(df[['diff']])
    df['cluster'] = km.labels_
    centers = km.cluster_centers_.flatten()
    normal_label = int(np.argmin(np.abs(centers)))
    is_outlier = df['cluster'] != normal_label
    for beat_t in np.arange(10, df['TIME(S)'].max() + 10, 10):
        low, high = beat_t - edge_sec, beat_t + edge_sec
        mask_window = (df['TIME(S)'] >= low) & (df['TIME(S)'] <= high)
        window_df = df[mask_window]
        if window_df.empty:
            continue
        left_val = window_df['R(OHM)'].iloc[0]
        right_val = window_df['R(OHM)'].iloc[-1]
        lo_b, hi_b = min(left_val, right_val), max(left_val, right_val)
        cond_outlier = (df['R(OHM)'] < lo_b) | (df['R(OHM)'] > hi_b)
        outlier_in_win = df[is_outlier & mask_window & cond_outlier]
        if outlier_in_win.empty:
            continue
        t_start = outlier_in_win['TIME(S)'].min()
        t_end = outlier_in_win['TIME(S)'].max()
        df = df[~((df['TIME(S)'] >= t_start) & (df['TIME(S)'] <= t_end))]
    return df.reset_index(drop=True)[['TIME(S)', 'R(OHM)']]


def split_normalize_detrend(df, period_length=20, air_sec=10, all_air=False):
    periods = []
    max_time = df['TIME(S)'].max()
    for start in np.arange(0, max_time, period_length):
        end = start + period_length
        period = df[(df['TIME(S)'] >= start) & (df['TIME(S)'] < end)].copy()
        if period.empty:
            continue
        actual_dur = period['TIME(S)'].max() - period['TIME(S)'].min()
        if actual_dur < period_length * 0.8:
            continue
        period['R(OHM)_norm'] = ((period['R(OHM)'] - period['R(OHM)'].min()) /
                                (period['R(OHM)'].max() - period['R(OHM)'].min()))
        if all_air:
            period['R(OHM)_detrended'] = period['R(OHM)_norm']
            periods.append(period)
            continue
        air_mask = period['TIME(S)'] < start + air_sec
        t_air = period.loc[air_mask, 'TIME(S)'].values
        y_air = period.loc[air_mask, 'R(OHM)_norm'].values
        if len(t_air) < 2:
            continue
        try:
            coeffs = np.polyfit(t_air - t_air[0], y_air, deg=1)
            trend = np.polyval(coeffs, period['TIME(S)'].values - t_air[0])
            period['R(OHM)_detrended'] = period['R(OHM)_norm'] - trend
            periods.append(period)
        except Exception:
            continue
    return periods


def align_time(period, target_len=500, sigma=1.0):
    t = period['TIME(S)'].values - period['TIME(S)'].iloc[0]
    y = period['R(OHM)_detrended'].values
    t_new = np.linspace(0, t[-1], target_len)
    y_new = np.interp(t_new, t, y)
    y_smooth = gaussian_filter1d(y_new, sigma=sigma)
    return y_smooth


def load_fold_models(model_dir, device):
    models = []
    for k in range(1, 6):
        m = CNN1D(input_length=500, num_classes=4).to(device)
        ckpt = torch.load(os.path.join(model_dir, f"fold{k}", "best_model.pth"),
                          map_location=device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def infer(models, X_np, device):
    X = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N,1,500)
    all_probs = []
    for m in models:
        logits = m(X)
        all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    ens = np.stack(all_probs, axis=0).mean(axis=0)  # (N,4)
    preds = ens.argmax(axis=1)
    return ens, preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    n_clusters, edge_sec, all_air = GAS_CFG[args.gas]
    print(f"gas={args.gas}  n_clusters={n_clusters}  edge_sec={edge_sec}  all_air={all_air}")

    df = read_csv(args.csv)
    print(f"raw points: {len(df)}, t_max={df['TIME(S)'].max():.1f}s")
    df = remove_anomaly(df, n_clusters, edge_sec)
    print(f"after anomaly removal: {len(df)} points")
    periods = split_normalize_detrend(df, all_air=all_air)
    print(f"valid 20s windows: {len(periods)}")
    if not periods:
        print("no valid windows")
        return
    X = np.stack([align_time(p) for p in periods], axis=0).astype(np.float32)
    print(f"X shape: {X.shape}")

    models = load_fold_models(args.model_dir, device)
    ens, preds = infer(models, X, device)

    print("\n===== per-window prediction =====")
    for i in range(len(preds)):
        p = CLASS_NAMES[preds[i]]
        s = "  ".join(f"{c}={ens[i,j]:.3f}" for j, c in enumerate(CLASS_NAMES))
        print(f"  win{i+1:02d}: pred={p:<4}  {s}")

    print("\n===== summary =====")
    dist = {c: int((preds == k).sum()) for k, c in enumerate(CLASS_NAMES)}
    print(f"prediction distribution: {dist}")
    print(f"per-class mean ensemble softmax: " +
          "  ".join(f"{c}={ens[:,k].mean():.3f}" for k, c in enumerate(CLASS_NAMES)))
    if args.gas in dist:
        n = len(preds)
        hit = dist[args.gas]
        print(f"correct as {args.gas}: {hit}/{n} ({hit/n*100:.1f}%)")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        out = pd.DataFrame(ens, columns=[f"p_{c}" for c in CLASS_NAMES])
        out.insert(0, "pred", [CLASS_NAMES[p] for p in preds])
        out.insert(0, "window", np.arange(1, len(preds)+1))
        out.to_csv(args.output_csv, index=False)
        print(f"\nper-window CSV -> {args.output_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--gas", required=True, choices=list(GAS_CFG.keys()))
    p.add_argument("--model_dir", default="results/cnn_seed42")
    p.add_argument("--output_csv", default=None)
    args = p.parse_args()
    main(args)
