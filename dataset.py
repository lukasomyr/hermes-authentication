"""
PyTorch Dataset with augmentation transforms.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),  # 20% of images converted to grayscale
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloaders():
    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=get_train_transforms())
    val_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=get_eval_transforms())
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=get_eval_transforms())

    print(f"Class mapping: {train_dataset.class_to_idx}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return {
        "train": train_loader, "val": val_loader, "test": test_loader,
        "train_dataset": train_dataset, "val_dataset": val_dataset, "test_dataset": test_dataset,
    }
