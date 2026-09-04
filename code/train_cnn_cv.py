"""1D-CNN training for gas discrimination (H2S / CH4 / CO2 / air).

Stratified five-fold cross-validation on the preprocessed 500-point
resistance-time waveforms. Saves per-fold best models, validation
classification reports, and the pooled out-of-fold (OOF) summary metrics.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
)

from model import CNN1D
from dataset import GasDataset

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_samples = 0.0, 0
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            probs.extend(F.softmax(logits, dim=1).cpu().numpy().tolist())
            labels.extend(yb.cpu().numpy().tolist())
    return total_loss / total_samples, np.array(labels), np.array(preds), np.array(probs)


def plot_confusion(y_true, y_pred, labels, fname):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def plot_loss_acc(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = np.arange(1, len(train_losses) + 1)
    train_losses_clipped = np.where(np.array(train_losses) > 2, np.nan, train_losses)
    val_losses_clipped = np.where(np.array(val_losses) > 2, np.nan, val_losses)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_losses_clipped, "r-", label="Train loss")
    ax1.plot(epochs, val_losses_clipped, "r--", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.set_ylim(0, 2)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accs, "b-", label="Train acc")
    ax2.plot(epochs, val_accs, "b--", label="Val acc")
    ax2.set_ylabel("Accuracy", color="b")
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.set_ylim(0, 1.0)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main(args):
    set_seed(args.seed)
    num_classes = len(CLASS_NAMES)
    X = np.load(os.path.join(args.data_dir, "X.npy"))
    y = np.load(os.path.join(args.data_dir, "y.npy"))
    print(f"X: {X.shape}, y: {y.shape}, "
          f"counts: {dict(zip(CLASS_NAMES, np.bincount(y)))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                            random_state=args.seed)

    base_dir = os.path.join(args.out_dir,
                            f"cnn_seed{args.seed}")
    os.makedirs(base_dir, exist_ok=True)

    all_y_true_per_fold, all_y_pred_per_fold = [], []
    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")
        fold_dir = os.path.join(base_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_loader = DataLoader(GasDataset(X_train, y_train, augment=True),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(GasDataset(X_val, y_val, augment=False),
                                batch_size=args.batch_size, shuffle=False)

        model = CNN1D(input_length=X.shape[1], num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15)

        best_val_acc, best_epoch = 0.0, 0
        best_model_path = os.path.join(fold_dir, "best_model.pth")
        history = {"epoch": [], "train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, running_correct, running_samples = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * yb.size(0)
                running_correct += (logits.argmax(1) == yb).sum().item()
                running_samples += yb.size(0)
            train_loss = running_loss / running_samples
            train_acc = running_correct / running_samples

            val_loss, y_true, y_pred, _ = evaluate(model, val_loader, device, criterion)
            val_acc = accuracy_score(y_true, y_pred)
            val_f1 = f1_score(y_true, y_pred, average="macro")
            scheduler.step(val_loss)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"  fold {fold} epoch {epoch}: train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                             "val_acc": val_acc}, best_model_path)
                print(f"  -> new best val_acc={best_val_acc:.4f}")
                plot_confusion(y_true, y_pred, CLASS_NAMES,
                               os.path.join(fold_dir, "confusion.png"))

            if epoch - best_epoch > args.patience:
                print(f"  early stopping at epoch {epoch} "
                      f"(best val_acc={best_val_acc:.4f} @ {best_epoch})")
                break

        plot_loss_acc(train_losses, val_losses, train_accs, val_accs,
                       os.path.join(fold_dir, "loss_acc.png"))
        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"),
                                      index=False)

        # reload best model and report on the held-out validation fold
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        _, y_true, y_pred, _ = evaluate(model, val_loader, device, criterion)
        all_y_true_per_fold.append(y_true)
        all_y_pred_per_fold.append(y_pred)
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                        output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(fold_dir, "classification_report.csv"))
        all_fold_metrics.append({"fold": fold, "best_val_acc": best_val_acc,
                                 "best_epoch": best_epoch, "val_f1": val_f1})

    # ----- pooled out-of-fold summary -----
    df_cv = pd.DataFrame(all_fold_metrics)
    df_cv.to_csv(os.path.join(base_dir, "cv_results.csv"), index=False)
    print("\n===== Cross-validation summary =====")
    print(df_cv)

    accs = [accuracy_score(t, p) for t, p in zip(all_y_true_per_fold, all_y_pred_per_fold)]
    f1s = [f1_score(t, p, average="macro") for t, p in zip(all_y_true_per_fold, all_y_pred_per_fold)]
    precisions = [precision_score(t, p, average="macro") for t, p in zip(all_y_true_per_fold, all_y_pred_per_fold)]
    recalls = [recall_score(t, p, average="macro") for t, p in zip(all_y_true_per_fold, all_y_pred_per_fold)]
    overall = {"Accuracy (%)": f"{np.mean(accs)*100:.1f} ± {np.std(accs)*100:.1f}",
               "F1-score": f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}",
               "Precision": f"{np.mean(precisions):.2f} ± {np.std(precisions):.2f}",
               "Recall": f"{np.mean(recalls):.2f} ± {np.std(recalls):.2f}"}
    pd.DataFrame([overall]).to_csv(os.path.join(base_dir, "overall_metrics.csv"),
                                    index=False)
    print("\n===== Overall metrics (OOF) =====")
    print(overall)

    # per-class OOF metrics
    yt = np.concatenate(all_y_true_per_fold)
    yp = np.concatenate(all_y_pred_per_fold)
    cls_report = classification_report(yt, yp, target_names=CLASS_NAMES,
                                        output_dict=True)
    per_class = [{"Class": c,
                  "Precision": cls_report[c]["precision"],
                  "Recall": cls_report[c]["recall"],
                  "F1-score": cls_report[c]["f1-score"]}
                 for c in CLASS_NAMES]
    pd.DataFrame(per_class).to_csv(
        os.path.join(base_dir, "per_class_metrics.csv"), index=False)
    print("\n===== Per-class metrics (OOF) =====")
    print(pd.DataFrame(per_class).to_string(index=False))

    # averaged (row-normalized) confusion matrix across folds
    cms = [confusion_matrix(t, p).astype(float) / confusion_matrix(t, p).sum(axis=1, keepdims=True)
           for t, p in zip(all_y_true_per_fold, all_y_pred_per_fold)]
    avg_cm = np.mean(cms, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=avg_cm,
                           display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues",
                                                              values_format=".2f")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "average_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"\nDone. Results saved under {base_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train 1D-CNN for gas discrimination.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--data_dir", type=str, default="data/preprocessed",
                   help="preprocessed .npy directory (X.npy, y.npy)")
    p.add_argument("--out_dir", type=str, default="results",
                   help="output root for run directories")
    args = p.parse_args()
    main(args)
