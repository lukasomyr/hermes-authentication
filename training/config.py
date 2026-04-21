"""
Central configuration for the Hermès Bag Authenticator.
"""

import os
import torch
import random
import numpy as np

# =============================================================
# ⚠️  UPDATE THIS PATH to match your Kaggle dataset
# =============================================================
RAW_DATA_DIR = "/kaggle/input/datasets/ignaciouriz/hermesbags/Dataset" # ← CHANGE THIS

PROCESSED_DATA_DIR = "/kaggle/working/data/processed"
OUTPUT_DIR = "/kaggle/working/outputs"

TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

# Data split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image
IMG_SIZE = 224
BACKGROUND_COLOR = (255, 255, 255)

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
FINE_TUNE_LR = 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 7
UNFREEZE_AFTER = 5

# Classes
CLASS_NAMES = ["Fake", "Real"]
NUM_CLASSES = 2

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
