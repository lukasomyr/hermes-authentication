"""
3-Fold Cross-Validation for the Hermès Authenticator.

Trains and evaluates the model 3 times on different splits,
then reports mean and std of all metrics.

Usage:
    python cross_validate.py
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

import training.config as config
from training.dataset import get_train_transforms, get_eval_transforms
from training.model import build_model, unfreeze_backbone, get_optimizer


NUM_FOLDS = 3


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="    Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="    Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_fold(model, train_loader, val_loader, device):
    """Train a single fold with the two-phase strategy."""

    # Class weights
    class_counts = [0] * config.NUM_CLASSES
    for _, label in train_loader.dataset:
        class_counts[label] += 1
    total_samples = sum(class_counts)
    class_weights = torch.tensor(
        [total_samples / c for c in class_counts], dtype=torch.float
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model, fine_tuning=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Phase 2: unfreeze backbone
        if epoch == config.UNFREEZE_AFTER + 1:
            print(f"    Unfreezing backbone at epoch {epoch}")
            unfreeze_backbone(model, num_blocks_to_unfreeze=3)
            optimizer = get_optimizer(model, fine_tuning=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"    Epoch {epoch:>2}/{config.NUM_EPOCHS}  "
              f"Train: {train_loss:.4f} / {train_acc:.4f}  "
              f"Val: {val_loss:.4f} / {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_state)
    return model


def main():
    config.seed_everything()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"  Hermès Authenticator — {NUM_FOLDS}-Fold Cross-Validation")
    print("=" * 60)
    print(f"  Device: {config.DEVICE}\n")

    # Load ALL processed images (combine train + val + test)
    all_data_dirs = [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]
    
    # Collect all image paths and labels
    all_images = []
    all_labels = []
    
    for data_dir in all_data_dirs:
        if not os.path.exists(data_dir):
            continue
        dataset = datasets.ImageFolder(root=data_dir)
        for path, label in dataset.samples:
            all_images.append(path)
            all_labels.append(label)
    
    all_labels = np.array(all_labels)
    print(f"  Total images: {len(all_images)}")
    print(f"  Class distribution: {dict(zip(config.CLASS_NAMES, np.bincount(all_labels)))}\n")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=config.SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels), 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{NUM_FOLDS}")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"{'='*60}")

        # Create datasets with appropriate transforms
        train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=get_train_transforms())
        val_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=get_eval_transforms())

        # Override the samples to use our fold split
        all_samples = []
        for data_dir in all_data_dirs:
            if os.path.exists(data_dir):
                ds = datasets.ImageFolder(root=data_dir)
                all_samples.extend(ds.samples)

        train_dataset.samples = [all_samples[i] for i in train_idx]
        train_dataset.targets = [all_labels[i] for i in train_idx]
        val_dataset.samples = [all_samples[i] for i in val_idx]
        val_dataset.targets = [all_labels[i] for i in val_idx]

        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE,
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=2, pin_memory=True
        )

        # Fresh model for each fold
        model = build_model(pretrained=True).to(config.DEVICE)
        model = train_fold(model, train_loader, val_loader, config.DEVICE)

        # Final evaluation on this fold's validation set
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, config.DEVICE)

        report = classification_report(
            labels, preds, target_names=config.CLASS_NAMES,
            digits=4, output_dict=True
        )

        fold_results.append({
            "fold": fold,
            "accuracy": val_acc,
            "fake_precision": report[config.CLASS_NAMES[0]]["precision"],
            "fake_recall": report[config.CLASS_NAMES[0]]["recall"],
            "fake_f1": report[config.CLASS_NAMES[0]]["f1-score"],
            "real_precision": report[config.CLASS_NAMES[1]]["precision"],
            "real_recall": report[config.CLASS_NAMES[1]]["recall"],
            "real_f1": report[config.CLASS_NAMES[1]]["f1-score"],
            "macro_f1": report["macro avg"]["f1-score"],
        })

        print(f"\n  Fold {fold} Results:")
        print(f"    Accuracy: {val_acc:.4f}")
        print(f"    Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(classification_report(labels, preds, target_names=config.CLASS_NAMES, digits=4))

    # ---- Summary across all folds ----
    results_df = pd.DataFrame(fold_results)
    
    print("\n" + "=" * 60)
    print(f"  CROSS-VALIDATION SUMMARY ({NUM_FOLDS} Folds)")
    print("=" * 60)
    
    metrics = ["accuracy", "fake_precision", "fake_recall", "fake_f1",
               "real_precision", "real_recall", "real_f1", "macro_f1"]
    
    summary_rows = []
    for metric in metrics:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        print(f"  {metric:<20s}: {mean:.4f} ± {std:.4f}")
        summary_rows.append({"metric": metric, "mean": mean, "std": std})
    
    # Save results
    results_df.to_csv(os.path.join(config.OUTPUT_DIR, "cv_fold_results.csv"), index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, "cv_summary.csv"), index=False)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-fold accuracy
    folds = results_df["fold"]
    axes[0].bar(folds, results_df["accuracy"], color=["#2196F3", "#4CAF50", "#FF9800"][:NUM_FOLDS])
    axes[0].axhline(y=results_df["accuracy"].mean(), color="red", linestyle="--",
                     label=f"Mean: {results_df['accuracy'].mean():.4f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy per Fold")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 scores comparison
    x = np.arange(NUM_FOLDS)
    width = 0.35
    axes[1].bar(x - width/2, results_df["fake_f1"], width, label="Fake F1", color="#F44336")
    axes[1].bar(x + width/2, results_df["real_f1"], width, label="Real F1", color="#2196F3")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score per Class per Fold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Fold {i+1}" for i in range(NUM_FOLDS)])
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, "cv_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {save_path}")
    print(f"  Results saved → {config.OUTPUT_DIR}/cv_fold_results.csv")
    print(f"  Summary saved → {config.OUTPUT_DIR}/cv_summary.csv")


if __name__ == "__main__":
    main()
