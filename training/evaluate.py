"""
Test set evaluation: classification report, confusion matrix, error analysis.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

import training.config as config
from training.dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from training.model import build_model


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs, all_images = [], [], [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_images.extend(images.cpu())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_images


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix - Test Set")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Confusion matrix -> {save_path}")


def plot_errors(images, labels, preds, probs, save_path, max_show=12):
    wrong = np.where(labels != preds)[0]
    if len(wrong) == 0:
        print("  No errors!"); return
    confs = [(i, probs[i][preds[i]]) for i in wrong]
    confs.sort(key=lambda x: -x[1])
    n = min(max_show, len(confs))
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.array(axes).flatten()
    for i, (idx, conf) in enumerate(confs[:n]):
        axes[i].imshow(denormalize(images[idx]))
        axes[i].set_title(f"True: {config.CLASS_NAMES[labels[idx]]}\nPred: {config.CLASS_NAMES[preds[idx]]} ({conf:.0%})", color="red", fontsize=10)
        axes[i].axis("off")
    for j in range(n, len(axes)): axes[j].axis("off")
    fig.suptitle(f"Misclassified ({len(wrong)} total errors)")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Error analysis -> {save_path}")


def main():
    config.seed_everything()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  Hermes Authenticator - Evaluation")
    print("=" * 60)

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model = model.to(config.DEVICE)

    data = get_dataloaders()
    preds, labels, probs, images = get_predictions(model, data["test"], config.DEVICE)

    print("\n--- Classification Report ---\n")
    report = classification_report(labels, preds, target_names=config.CLASS_NAMES, digits=4)
    print(report)

    with open(os.path.join(config.OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_confusion_matrix(labels, preds, os.path.join(config.OUTPUT_DIR, "confusion_matrix.png"))
    plot_errors(images, labels, preds, probs, os.path.join(config.OUTPUT_DIR, "error_analysis.png"))

    acc = (preds == labels).mean()
    print(f"\n  Test Accuracy: {acc:.4f} ({acc:.1%})")
    print(f"  Misclassified: {(preds != labels).sum()} / {len(labels)}")

if __name__ == "__main__":
    main()
