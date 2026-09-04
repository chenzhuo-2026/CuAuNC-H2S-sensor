"""Build the unseen-concentration H2S evaluation (Table S5).

Freezes five pure-H2S source recordings at concentrations excluded from the
original 320-sample training set (20 ppb, 50 ppb, 50 ppm, 100 ppm, 150 ppm),
runs the frozen seed-42 five-fold CNN ensemble on each 10-window recording,
and exports a 50-row CSV (with source_csv + source_sha256 for traceability).
Asserts 44/50 H2S-correct to match the response letter. Requires the raw
recordings under raw_root and the infer_unseen_h2s helper module.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_unseen_h2s as isc  # noqa: E402  (same-dir helper)

# frozen canonical H2S source recordings (concentration -> relative path)
CANONICAL = [
    ("20ppb",  "unseen_H2S/20ppb.csv"),
    ("50ppb",  "unseen_H2S/50ppb.csv"),
    ("50ppm",  "unseen_H2S/50ppm.csv"),
    ("100ppm", "unseen_H2S/100ppm.csv"),
    ("150ppm", "unseen_H2S/150ppm.csv"),
]
CLASS_NAMES = isc.CLASS_NAMES


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = isc.load_fold_models(args.model_dir, device)
    n_clusters, edge_sec, all_air = isc.GAS_CFG["H2S"]

    rows = []
    print(f"model: {args.model_dir} (frozen seed-42 5-fold ensemble)")
    print(f"preprocess: n_clusters={n_clusters}, edge_sec={edge_sec}, all_air={all_air}\n")
    for conc, rel in CANONICAL:
        path = os.path.join(args.raw_root, rel)
        assert os.path.exists(path), f"MISSING: {path}"
        s = sha256(path)
        df = isc.read_csv(path)
        df = isc.remove_anomaly(df, n_clusters, edge_sec)
        periods = isc.split_normalize_detrend(df, all_air=all_air)
        assert len(periods) == 10, f"{conc}: expected 10 windows, got {len(periods)}"
        X = np.stack([isc.align_time(p) for p in periods]).astype(np.float32)
        ens, preds = isc.infer(models, X, device)
        h2s = int((preds == 0).sum())
        for w in range(len(preds)):
            rows.append({
                "concentration": conc, "source_csv": os.path.basename(path),
                "source_path": path, "source_sha256": s, "window": w + 1,
                "pred": CLASS_NAMES[preds[w]],
                "p_H2S": round(float(ens[w, 0]), 4),
                "p_CH4": round(float(ens[w, 1]), 4),
                "p_CO2": round(float(ens[w, 2]), 4),
                "p_AIR": round(float(ens[w, 3]), 4),
                "correct_H2S": int(preds[w] == 0)})
        print(f"  {conc:<7} {os.path.basename(path):<14} sha256={s[:12]}...  {h2s}/10 H2S")

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    # traceability assertions
    total = len(df_out)
    total_h2s = int(df_out["correct_H2S"].sum())
    assert total == 50
    assert all(df_out.groupby("concentration").size() == 10)
    assert len(df_out["source_sha256"].unique()) == 5
    for _, r in df_out.iterrows():
        ps = [r["p_H2S"], r["p_CH4"], r["p_CO2"], r["p_AIR"]]
        assert CLASS_NAMES[ps.index(max(ps))] == r["pred"]
    print(f"\n-> {args.out_csv} ({total} rows); per-conc: "
          f"{df_out.groupby('concentration')['correct_H2S'].sum().to_dict()}")
    print(f"TOTAL: {total_h2s}/50 H2S")
    assert total_h2s == 44, (
        f"canonical total is {total_h2s}/50, NOT 44 (response letter says 44). "
        "Either the source files changed or the response must be updated.")
    print("assert total_h2s == 44: PASSED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", type=str,
                   default="results/cnn_seed42")
    p.add_argument("--raw_root", type=str, default="data/raw_data",
                   help="root dir of the raw H2S recordings")
    p.add_argument("--out_csv", type=str,
                   default="results/unseen_concentration/unseen_h2s_predictions.csv")
    args = p.parse_args()
    main(args)
