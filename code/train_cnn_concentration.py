"""1D-CNN training under the concentration-level split (Table S4 CNN row).

Five-fold CV on the training portion of the held-out split, with
class-balanced cross-entropy (FocalLoss available but off by default),
then final evaluation on the fixed test set: per-fold single-model
metrics (mean +/- SD, the reported 89.7 +/- 0.8%) and the five-fold
soft-voting ensemble.
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
from sklearn.metrics import (accuracy_score, classification_report,
                                confusion_matrix, ConfusionMatrixDisplay,
                                f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

from model import CNN1D
from dataset import GasDataset

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]


class FocalLoss(nn.Module):
    """Focal loss with optional per-class weights (alpha) and focusing gamma."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


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


def plot_confusion(y_true, y_pred, labels, fname, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred),
                            display_labels=labels).plot(ax=ax, cmap="Blues",
                                                                  values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def plot_loss_acc(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = np.arange(1, len(train_losses) + 1)
    tl = np.where(np.array(train_losses) > 2, np.nan, train_losses)
    vl = np.where(np.array(val_losses) > 2, np.nan, val_losses)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, tl, "r-", label="Train loss")
    ax1.plot(epochs, vl, "r--", label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="r")
    ax1.tick_params(axis="y", labelcolor="r"); ax1.set_ylim(0, 2)
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accs, "b-", label="Train acc")
    ax2.plot(epochs, val_accs, "b--", label="Val acc")
    ax2.set_ylabel("Accuracy", color="b")
    ax2.tick_params(axis="y", labelcolor="b"); ax2.set_ylim(0, 1.0)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    plt.grid(True); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close(fig)


def main(args):
    set_seed(args.seed)
    num_classes = len(CLASS_NAMES)
    data_dir = args.data_dir or "data/concentration_split"
    if not os.path.exists(data_dir):
        data_dir = "split"
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    base_dir = os.path.join(args.out_dir,
                            f"cnn_seed{args.seed}")
    os.makedirs(base_dir, exist_ok=True)

    all_y_true, all_y_pred = [], []
    all_fold_metrics = []

    # five-fold CV on the training portion
    for fold, (tr, va) in enumerate(kfold.split(X_train, y_train), 1):
        print(f"\n===== Fold {fold} =====")
        fold_dir = os.path.join(base_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        X_tr, X_va = X_train[tr], X_train[va]
        y_tr, y_va = y_train[tr], y_train[va]
        train_loader = DataLoader(GasDataset(X_tr, y_tr, augment=True),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(GasDataset(X_va, y_va, augment=False),
                                batch_size=args.batch_size, shuffle=False)

        model = CNN1D(input_length=X_train.shape[1], num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                                 factor=0.5, patience=15)
        epochs, patience = 200, 30
        best_val_acc, best_epoch = 0.0, 0
        best_model_path = os.path.join(fold_dir, "best_model.pth")
        history = {"epoch": [], "train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}
        tl, vl, ta, va_acc = [], [], [], []

        # class-balanced loss (FocalLoss available via --focal)
        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        weights = torch.tensor(cw, dtype=torch.float32).to(device)
        criterion = (FocalLoss(alpha=weights, gamma=2.0) if args.focal
                     else nn.CrossEntropyLoss(weight=weights))

        for epoch in range(1, epochs + 1):
            model.train()
            rl, rc, rs = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward(); optimizer.step()
                rl += loss.item() * yb.size(0)
                rc += (logits.argmax(1) == yb).sum().item()
                rs += yb.size(0)
            train_loss, train_acc = rl / rs, rc / rs
            val_loss, yt, yp, _ = evaluate(model, val_loader, device, criterion)
            val_acc = accuracy_score(yt, yp)
            val_f1 = f1_score(yt, yp, average="macro")
            scheduler.step(val_loss)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            tl.append(train_loss); vl.append(val_loss)
            ta.append(train_acc); va_acc.append(val_acc)
            print(f"  fold {fold} epoch {epoch}: train_acc={train_acc:.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                             "optimizer_state": optimizer.state_dict(),
                             "val_acc": val_acc}, best_model_path)
                plot_confusion(yt, yp, CLASS_NAMES,
                                os.path.join(fold_dir, "confusion_val.png"),
                                f"fold {fold} validation")
            if epoch - best_epoch > patience:
                print(f"  early stop @ {epoch} (best val_acc={best_val_acc:.4f} @ {best_epoch})")
                break

        plot_loss_acc(tl, vl, ta, va_acc, os.path.join(fold_dir, "loss_acc.png"))
        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        _, yt, yp, _ = evaluate(model, val_loader, device, criterion)
        all_y_true.append(yt); all_y_pred.append(yp)
        pd.DataFrame(classification_report(yt, yp, target_names=CLASS_NAMES,
                                            output_dict=True)).transpose().to_csv(
            os.path.join(fold_dir, "classification_report_val.csv"))
        all_fold_metrics.append({"fold": fold, "best_val_acc": best_val_acc,
                                 "best_epoch": best_epoch, "val_f1": val_f1})

    # validation summary
    pd.DataFrame(all_fold_metrics).to_csv(os.path.join(base_dir, "cv_results_val.csv"),
                                            index=False)
    accs = [accuracy_score(t, p) for t, p in zip(all_y_true, all_y_pred)]
    f1s = [f1_score(t, p, average="macro") for t, p in zip(all_y_true, all_y_pred)]
    precs = [precision_score(t, p, average="macro") for t, p in zip(all_y_true, all_y_pred)]
    recs = [recall_score(t, p, average="macro") for t, p in zip(all_y_true, all_y_pred)]
    val_metrics = {"Accuracy (%)": f"{np.mean(accs)*100:.1f} +/- {np.std(accs)*100:.1f}",
                   "F1-score": f"{np.mean(f1s):.2f} +/- {np.std(f1s):.2f}",
                   "Precision": f"{np.mean(precs):.2f} +/- {np.std(precs):.2f}",
                   "Recall": f"{np.mean(recs):.2f} +/- {np.std(recs):.2f}"}
    pd.DataFrame([val_metrics]).to_csv(os.path.join(base_dir, "overall_metrics_val.csv"),
                                         index=False)
    print(f"\nvalidation: {val_metrics}")

    # final evaluation on the held-out test set
    test_loader = DataLoader(GasDataset(X_test, y_test), batch_size=args.batch_size,
                              shuffle=False)
    all_fold_test_probs = []
    for fold in range(1, 6):
        m = CNN1D(input_length=X_train.shape[1], num_classes=num_classes).to(device)
        m.load_state_dict(torch.load(os.path.join(base_dir, f"fold{fold}", "best_model.pth"),
                                       map_location=device)["model_state"])
        m.eval()
        _, _, _, probs = evaluate(m, test_loader, device, criterion)
        all_fold_test_probs.append(probs)

    # per-fold single-model test metrics (mean +/- SD) — the reported 89.7
    fold_accs, fold_f1s, fold_precs, fold_recs = [], [], [], []
    for probs in all_fold_test_probs:
        p = probs.argmax(1)
        fold_accs.append(accuracy_score(y_test, p))
        fold_f1s.append(f1_score(y_test, p, average="macro"))
        fold_precs.append(precision_score(y_test, p, average="macro"))
        fold_recs.append(recall_score(y_test, p, average="macro"))
    test_metrics = {"Accuracy (%)": f"{np.mean(fold_accs)*100:.1f} +/- {np.std(fold_accs)*100:.1f}",
                    "F1-score": f"{np.mean(fold_f1s):.2f} +/- {np.std(fold_f1s):.2f}",
                    "Precision": f"{np.mean(fold_precs):.2f} +/- {np.std(fold_precs):.2f}",
                    "Recall": f"{np.mean(fold_recs):.2f} +/- {np.std(fold_recs):.2f}"}
    pd.DataFrame([test_metrics]).to_csv(os.path.join(base_dir, "overall_metrics_test.csv"),
                                          index=False)
    print(f"test (5-fold single, mean+/-SD): {test_metrics}")

    # five-fold soft-voting ensemble
    ens_probs = np.mean(all_fold_test_probs, axis=0)
    ens_pred = ens_probs.argmax(1)
    ens_metrics = {"Accuracy (%)": f"{accuracy_score(y_test, ens_pred)*100:.1f}",
                   "F1-score": f"{f1_score(y_test, ens_pred, average='macro'):.2f}",
                   "Precision": f"{precision_score(y_test, ens_pred, average='macro'):.2f}",
                   "Recall": f"{recall_score(y_test, ens_pred, average='macro'):.2f}"}
    pd.DataFrame([ens_metrics]).to_csv(os.path.join(base_dir, "ensemble_final_metrics.csv"),
                                          index=False)
    pd.DataFrame(classification_report(y_test, ens_pred, target_names=CLASS_NAMES,
                                         output_dict=True)).transpose().to_csv(
        os.path.join(base_dir, "classification_report_ensemble_test.csv"))
    np.save(os.path.join(base_dir, "ensemble_probs_test.npy"), ens_probs)
    np.save(os.path.join(base_dir, "ensemble_preds_test.npy"), ens_pred)
    plot_confusion(y_test, ens_pred, CLASS_NAMES,
                    os.path.join(base_dir, "confusion_ensemble_test.png"),
                    "ensemble test set")
    print(f"ensemble: {ens_metrics}")
    print(f"\n-> {base_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--focal", action="store_true", help="use FocalLoss instead of CE")
    p.add_argument("--data_dir", type=str, default=None,
                   help="split dir (default data/concentration_split)")
    p.add_argument("--out_dir", type=str, default="results/cnn_concentration")
    args = p.parse_args()
    main(args)
