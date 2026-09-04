"""KNN baseline (five-fold CV, Table S3 KNN row).

Runs only KNN (does not overwrite the RF/SVM/XGBoost results). Uses the
preprocessed data with StratifiedKFold(5, shuffle, random_state=seed).
KNN pipeline: StandardScaler + KNeighborsClassifier(k=5), scaler fit
within-fold (no leakage). Saves per-fold metrics, the saved KNN models,
and an OOF prediction table (array_index + fold) for independent recompute.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train_ml_cv import extract_features


def main(args):
    np.random.seed(args.seed)
    X = np.load(os.path.join(args.data_dir, "X.npy"))
    y = np.load(os.path.join(args.data_dir, "y.npy"))
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    base = Pipeline([("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=args.k))])

    os.makedirs(args.out_dir, exist_ok=True)
    accs, f1s, precs, recs = [], [], [], []
    oof_rows = []

    for fold, (tr, va) in enumerate(kfold.split(X, y), 1):
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]
        X_tr_feat = extract_features(X_tr, augment=True)
        X_va_feat = extract_features(X_va, augment=False)
        m = clone(base)
        m.fit(X_tr_feat, y_tr)
        y_pred = m.predict(X_va_feat)
        joblib.dump(m, os.path.join(args.out_dir, f"KNN_fold{fold}.joblib"))
        accs.append(accuracy_score(y_va, y_pred))
        f1s.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
        precs.append(precision_score(y_va, y_pred, average="macro", zero_division=0))
        recs.append(recall_score(y_va, y_pred, average="macro", zero_division=0))
        for i, idx in enumerate(va):
            oof_rows.append({"array_index": int(idx), "fold": fold,
                             "y_true": int(y_va[i]), "y_pred": int(y_pred[i])})
        print(f"fold {fold}: acc={accs[-1]:.4f} f1={f1s[-1]:.4f}")

    pd.DataFrame({"fold": range(1, 6), "accuracy": accs,
                  "macro_precision": precs, "macro_recall": recs,
                  "macro_f1": f1s}).to_csv(
        os.path.join(args.out_dir, "KNN_fold_metrics.csv"), index=False)

    oof_df = pd.DataFrame(oof_rows)
    assert sorted(oof_df["array_index"].tolist()) == list(range(len(X))), "OOF coverage mismatch"
    assert set(oof_df["fold"].unique()) == {1, 2, 3, 4, 5}, "fold mismatch"
    assert len(oof_df) == len(X), "OOF count mismatch"
    oof_df.to_csv(os.path.join(args.out_dir, "KNN_oof_predictions.csv"), index=False)

    # independent recompute from OOF
    for f in range(1, 6):
        sub = oof_df[oof_df.fold == f]
        assert abs(accuracy_score(sub["y_true"], sub["y_pred"]) - accs[f - 1]) < 1e-9
    print("OOF independent recompute: PASS")

    summary = pd.DataFrame([{
        "Model": "KNN",
        "Accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs, ddof=0):.4f}",
        "F1-Score": f"{np.mean(f1s):.4f} +/- {np.std(f1s, ddof=0):.4f}",
        "Precision": f"{np.mean(precs):.4f} +/- {np.std(precs, ddof=0):.4f}",
        "Recall": f"{np.mean(recs):.4f} +/- {np.std(recs, ddof=0):.4f}"}])
    summary.to_csv(args.out_csv, index=False)
    print(f"KNN: acc={np.mean(accs):.4f} +/- {np.std(accs, ddof=0):.4f}")
    print(f"-> {args.out_csv} + {args.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default="data/preprocessed")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out_dir", type=str, default="results/ml")
    p.add_argument("--out_csv", type=str, default="results/ml/knn_baseline_results.csv")
    args = p.parse_args()
    main(args)
