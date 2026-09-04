"""KNN baseline evaluation under the concentration-level split (Table S4).

Pipeline: load the held-out split (concentration_split), run 5-fold CV on
the training portion only, train KNN (StandardScaler + KNeighborsClassifier,
k=5, scaler fit within-fold to avoid leakage) on each fold, and evaluate on
the fixed held-out test set; report mean +/- SD across the 5 fold-specific
models.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train_ml_cv import extract_features


def main(args):
    np.random.seed(args.seed)
    d = args.data_dir
    X_train = np.load(os.path.join(d, "X_train.npy"))
    y_train = np.load(os.path.join(d, "y_train.npy"))
    X_test = np.load(os.path.join(d, "X_test.npy"))
    y_test = np.load(os.path.join(d, "y_test.npy"))
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    X_test_feat = extract_features(X_test, augment=False)

    fold_metrics = []
    for fold, (tr_idx, _) in enumerate(kfold.split(X_train, y_train), 1):
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_tr_feat = extract_features(X_tr, augment=True)
        model = Pipeline([("scaler", StandardScaler()),
                          ("clf", KNeighborsClassifier(n_neighbors=args.k))])
        model.fit(X_tr_feat, y_tr)
        pred = model.predict(X_test_feat)
        fold_metrics.append({
            "fold": fold, "accuracy": accuracy_score(y_test, pred),
            "f1_macro": f1_score(y_test, pred, average="macro"),
            "precision_macro": precision_score(y_test, pred, average="macro"),
            "recall_macro": recall_score(y_test, pred, average="macro")})
        print(f"  fold {fold}: acc={fold_metrics[-1]['accuracy']:.4f}")

    df = pd.DataFrame(fold_metrics)
    print("\n=== KNN (concentration-level split) ===")
    print(f"Accuracy:  {df['accuracy'].mean()*100:.1f} +/- {np.std(df['accuracy'].values)*100:.1f}%")
    print(f"F1-score:  {df['f1_macro'].mean():.2f} +/- {np.std(df['f1_macro'].values):.2f}")
    print(f"Precision: {df['precision_macro'].mean():.2f} +/- {np.std(df['precision_macro'].values):.2f}")
    print(f"Recall:    {df['recall_macro'].mean():.2f} +/- {np.std(df['recall_macro'].values):.2f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, default="data/concentration_split",
                   help="dir with X_train/y_train/X_test/y_test .npy")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out_csv", type=str,
                   default="results/ml_concentration/knn_s4_test_results.csv")
    args = p.parse_args()
    main(args)
