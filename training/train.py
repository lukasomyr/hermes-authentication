"""
training/train.py
Fine-tunes a pretrained YOLOv8 model on the SKU-110K dataset.

Usage:
    python training/train.py
"""

from ultralytics import YOLO
import yaml
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data/SKU110K"
YAML_PATH  = "training/sku110k.yaml"
MODEL      = "yolov8n.pt"   # nano = fast; swap for yolov8s.pt for better accuracy
EPOCHS     = 30
IMG_SIZE   = 640
BATCH      = 16
PROJECT    = "runs/train"
RUN_NAME   = "sku110k_finetune"
# ──────────────────────────────────────────────────────────────────────────────


def create_yaml():
    """Creates the dataset YAML config expected by Ultralytics."""
    cfg = {
        "path": os.path.abspath(DATA_DIR),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": ["product"],
    }
    os.makedirs("training", exist_ok=True)
    with open(YAML_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Dataset YAML written to {YAML_PATH}")


def train():
    create_yaml()

    model = YOLO(MODEL)
    print(f"Starting fine-tuning: {MODEL} → {EPOCHS} epochs")

    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=RUN_NAME,
        pretrained=True,
        patience=10,        # early stopping
        save=True,
        plots=True,
    )

    print("\nTraining complete!")
    print(f"Best weights saved to: {PROJECT}/{RUN_NAME}/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
