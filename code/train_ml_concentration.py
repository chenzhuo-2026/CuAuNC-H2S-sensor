"""Conventional ML baselines under the concentration-level split (Table S4).

Trains RF / linear SVM / XGBoost on the 9 handcrafted descriptors, using
five-fold CV on the training portion only and final evaluation on the held-out
test set (mean +/- SD across the five fold-specific models), plus a soft-voting
ensemble. Output dir defaults to results/ml_concentration.
"""
import argparse
import os

import joblib
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
import matplotlib.pyplot as plt

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]


def augment_signal(x, noise_std=0.01, crop_ratio=0.1):
    x_aug = x.copy()
    if np.random.rand() < 0.5:
        x_aug = x_aug + np.random.randn(*x_aug.shape) * noise_std
    if np.random.rand() < 0.5:
        crop_len = int((1 - crop_ratio) * len(x_aug))
        start = np.random.randint(0, len(x_aug) - crop_len + 1)
        cropped = x_aug[start:start + crop_len]
        x_aug = np.concatenate([cropped, np.zeros(len(x_aug) - len(cropped))])
    return x_aug


def extract_features(X, augment=False):
    """Nine descriptors: six time-domain stats + three frequency-domain stats
    of the one-sided FFT magnitude."""
    features = []
    for x in X:
        if augment:
            x = augment_signal(x)
        feat = [np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x), np.ptp(x)]
        fft_power = np.abs(np.fft.fft(x)[:len(x) // 2])
        feat += [np.mean(fft_power), np.std(fft_power), np.max(fft_power)]
        features.append(feat)
    return np.array(features)


def plot_confusion(y_true, y_pred, fname, title=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred),
                            display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues",
                                                                  values_format="d")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def main(args):
    np.random.seed(args.seed)
    out_dir = args.output_dir or f"results/ml_concentration"
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    data_dir = args.data_dir or f"data/concentration_split"
    if not os.path.exists(data_dir):
        data_dir = "split"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=args.seed),
        "SVM_(linear)": SVC(kernel="linear", C=1.0, probability=True, random_state=args.seed),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric="mlogloss", random_state=args.seed),
    }
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # five-fold CV on training set
    for name, base in models.items():
        print(f"\n===== {name} =====")
        fold_metrics = []
        for fold, (tr, va) in enumerate(kfold.split(X_train, y_train), 1):
            X_tr_feat = extract_features(X_train[tr], augment=True)
            X_val_feat = extract_features(X_train[va], augment=False)
            m = clone(base)
            m.fit(X_tr_feat, y_train[tr])
            joblib.dump(m, os.path.join(models_dir, f"{name}_fold{fold}.joblib"))
            pred = m.predict(X_val_feat)
            fold_metrics.append({"fold": fold, "accuracy": accuracy_score(y_train[va], pred),
                                 "f1_macro": f1_score(y_train[va], pred, average="macro"),
                                 "precision_macro": precision_score(y_train[va], pred, average="macro"),
                                 "recall_macro": recall_score(y_train[va], pred, average="macro")})
            fd = os.path.join(out_dir, name.replace("(", "").replace(")", ""), f"fold{fold}")
            os.makedirs(fd, exist_ok=True)
            pd.DataFrame(classification_report(y_train[va], pred, target_names=CLASS_NAMES,
                                                output_dict=True)).transpose().to_csv(
                os.path.join(fd, "classification_report_val.csv"))
            plot_confusion(y_train[va], pred, os.path.join(fd, "confusion_matrix_val.png"),
                           f"{name} fold {fold} val")
        pd.DataFrame(fold_metrics).to_csv(os.path.join(out_dir, f"{name}_fold_results_val.csv"),
                                            index=False)
        print(f"  CV acc={np.mean([m['accuracy'] for m in fold_metrics]):.4f}")

    # evaluate on held-out test set (per-fold + ensemble)
    test_rows = {}
    for name in models:
        per_fold = []
        all_probs = []
        for fold in range(1, 6):
            m = joblib.load(os.path.join(models_dir, f"{name}_fold{fold}.joblib"))
            X_test_feat = extract_features(X_test, augment=False)
            probs = m.predict_proba(X_test_feat)
            all_probs.append(probs)
            pred = probs.argmax(1)
            per_fold.append({"fold": fold, "accuracy": accuracy_score(y_test, pred),
                             "f1_macro": f1_score(y_test, pred, average="macro"),
                             "precision_macro": precision_score(y_test, pred, average="macro"),
                             "recall_macro": recall_score(y_test, pred, average="macro")})
        df = pd.DataFrame(per_fold)
        df.to_csv(os.path.join(out_dir, f"{name}_test_results.csv"), index=False)
        test_rows[name] = {"Accuracy": f"{df['accuracy'].mean():.4f} +/- {np.std(df['accuracy'].values):.4f}",
                           "F1-Score": f"{df['f1_macro'].mean():.4f} +/- {np.std(df['f1_macro'].values):.4f}"}
        print(f"  {name} test acc={df['accuracy'].mean():.4f} +/- {np.std(df['accuracy'].values):.4f}")

        # ensemble (soft vote)
        ens_pred = np.mean(all_probs, axis=0).argmax(1)
        ens_dir = os.path.join(out_dir, name.replace("(", "").replace(")", ""), "Ensemble")
        os.makedirs(ens_dir, exist_ok=True)
        report = classification_report(y_test, ens_pred, target_names=CLASS_NAMES, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(ens_dir, "classification_report_test.csv"))
        np.save(os.path.join(ens_dir, "ensemble_probs_test.npy"), np.mean(all_probs, axis=0))
        np.save(os.path.join(ens_dir, "ensemble_preds_test.npy"), ens_pred)

    pd.DataFrame(test_rows).transpose().to_csv(os.path.join(out_dir, "test_set_summary.csv"))
    print(f"\n-> {out_dir}/test_set_summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data_dir", type=str, default=None,
                   help="split dir (default concentration_split)")
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()
    main(args)
