"""Conventional ML baselines (RF / linear SVM / XGBoost) under five-fold CV.

Feeds the same 320-sample dataset as the CNN, but uses nine predefined
handcrafted descriptors per window (six time-domain statistics + three
frequency-domain statistics from the one-sided FFT magnitude) instead of
the raw waveform. Training-fold augmentation (Gaussian noise sigma=0.01
and 10% random crop, each p=0.5) matches the CNN protocol. Results are
not directly comparable to the CNN at the classifier level because the
input representation also differs; the comparison is therefore
pipeline-level.

Note: extract_features() and augment_signal() are shared with the KNN
baseline scripts (train_knn_cv.py, eval_knn_concentration.py).
"""
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                                confusion_matrix, ConfusionMatrixDisplay,
                                f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]


def augment_signal(x, noise_std=0.01, crop_ratio=0.1):
    """Training-time augmentation: Gaussian noise + random crop (p=0.5 each)."""
    x_aug = x.copy()
    if np.random.rand() < 0.5:
        x_aug = x_aug + np.random.randn(*x_aug.shape) * noise_std
    if np.random.rand() < 0.5:
        crop_len = int((1 - crop_ratio) * len(x_aug))
        start = np.random.randint(0, len(x_aug) - crop_len + 1)
        cropped = x_aug[start:start + crop_len]
        pad_len = len(x_aug) - len(cropped)
        x_aug = np.concatenate([cropped, np.zeros(pad_len)])
    return x_aug


def extract_features(X, augment=False):
    """Nine descriptors per window: six time-domain statistics + three
    frequency-domain statistics of the one-sided FFT magnitude."""
    features = []
    for x in X:
        if augment:
            x = augment_signal(x)
        feat = [np.mean(x), np.std(x), np.min(x), np.max(x),
                np.median(x), np.ptp(x)]
        fft_power = np.abs(np.fft.fft(x)[:len(x) // 2])
        feat += [np.mean(fft_power), np.std(fft_power), np.max(fft_power)]
        features.append(feat)
    return np.array(features)


def main(args):
    np.random.seed(args.seed)
    print(f"ML baseline training - seed: {args.seed}")

    X = np.load(os.path.join(args.data_dir, "X.npy"))
    y = np.load(os.path.join(args.data_dir, "y.npy"))

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=1),
        "SVM (linear)": SVC(kernel="linear", C=1.0, probability=True, random_state=args.seed),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=args.seed, n_jobs=1),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    summary = []

    for name, base_model in models.items():
        accs, f1s, precs, recs = [], [], [], []
        all_y_true, all_y_pred = [], []
        print(f"\n===== {name} =====")
        for fold, (tr, va) in enumerate(kfold.split(X, y), 1):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]
            X_tr_feat = extract_features(X_tr, augment=True)
            X_va_feat = extract_features(X_va, augment=False)
            model = clone(base_model)
            model.fit(X_tr_feat, y_tr)
            y_pred = model.predict(X_va_feat)
            joblib.dump(model, os.path.join(args.out_dir,
                                              f"{name.replace(' ', '_')}_fold{fold}.joblib"))
            accs.append(accuracy_score(y_va, y_pred))
            f1s.append(f1_score(y_va, y_pred, average="macro"))
            precs.append(precision_score(y_va, y_pred, average="macro"))
            recs.append(recall_score(y_va, y_pred, average="macro"))
            all_y_true.append(y_va)
            all_y_pred.append(y_pred)
            report = classification_report(y_va, y_pred, target_names=CLASS_NAMES,
                                            output_dict=True)
            pd.DataFrame(report).to_csv(os.path.join(
                args.out_dir, f"classification_report_{name.replace(' ', '_')}_fold{fold}.csv"))
            cm = confusion_matrix(y_va, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
                ax=ax, cmap="Blues", values_format="d")
            plt.title(f"{name} fold {fold}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir,
                                       f"cm_{name.replace(' ', '_')}_fold{fold}.png"), dpi=150)
            plt.close(fig)

        print(f"  acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}  "
              f"f1={np.mean(f1s):.4f}  prec={np.mean(precs):.4f}  rec={np.mean(recs):.4f}")
        summary.append({"Model": name,
                        "Accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
                        "F1-Score": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
                        "Precision": f"{np.mean(precs):.4f} +/- {np.std(precs):.4f}",
                        "Recall": f"{np.mean(recs):.4f} +/- {np.std(recs):.4f}"})

        # overall (pooled OOF) confusion matrix
        yt = np.concatenate(all_y_true)
        yp = np.concatenate(all_y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(yt, yp),
                                display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues",
                                                                    values_format="d")
        plt.title(f"{name} overall confusion matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"cm_{name.replace(' ', '_')}_overall.png"),
                     dpi=150)
        plt.close(fig)
        pd.DataFrame(summary).to_csv(args.out_csv, index=False)

    print("\n===== Baseline summary =====")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"-> {args.out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--data_dir", type=str, default="data/preprocessed")
    p.add_argument("--out_dir", type=str, default="results/ml")
    p.add_argument("--out_csv", type=str, default="results/ml/baseline_summary.csv")
    args = p.parse_args()
    main(args)
