"""
Training with two-phase transfer learning + early stopping.
"""

import os, time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import training.config as config
from training.dataset import get_dataloaders
from training.model import build_model, unfreeze_backbone, get_optimizer


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Training", leave=False):
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
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total


def plot_curves(log_df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train", marker="o", ms=3)
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="Val", marker="o", ms=3)
    if config.UNFREEZE_AFTER < log_df["epoch"].max():
        axes[0].axvline(x=config.UNFREEZE_AFTER, color="gray", ls="--", alpha=0.7, label="Unfreeze")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(log_df["epoch"], log_df["train_acc"], label="Train", marker="o", ms=3)
    axes[1].plot(log_df["epoch"], log_df["val_acc"], label="Val", marker="o", ms=3)
    if config.UNFREEZE_AFTER < log_df["epoch"].max():
        axes[1].axvline(x=config.UNFREEZE_AFTER, color="gray", ls="--", alpha=0.7, label="Unfreeze")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Curves saved -> {save_path}")


def main():
    config.seed_everything()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Hermes Authenticator - Training")
    print("=" * 60)
    print(f"  Device: {config.DEVICE}")

    data = get_dataloaders()
    train_loader, val_loader = data["train"], data["val"]

    # Class weights for imbalance
    train_dataset = data["train_dataset"]
    class_counts = [0] * config.NUM_CLASSES
    for _, label in train_dataset:
        class_counts[label] += 1
    total_samples = sum(class_counts)
    class_weights = torch.tensor([total_samples / c for c in class_counts], dtype=torch.float).to(config.DEVICE)
    print(f"Class counts: {dict(zip(config.CLASS_NAMES, class_counts))}")

    model = build_model(pretrained=True).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model, fine_tuning=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    log_rows = []
    start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")

        if epoch == config.UNFREEZE_AFTER + 1:
            print("\n*** Phase 2: Unfreezing backbone ***")
            unfreeze_backbone(model, num_blocks_to_unfreeze=3)
            optimizer = get_optimizer(model, fine_tuning=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"  Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.PATIENCE})")
            if patience_counter >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nDone in {(time.time()-start)/60:.1f} min. Best val_loss: {best_val_loss:.4f}")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(os.path.join(config.OUTPUT_DIR, "training_log.csv"), index=False)
    plot_curves(log_df, os.path.join(config.OUTPUT_DIR, "training_curves.png"))

if __name__ == "__main__":
    main()
