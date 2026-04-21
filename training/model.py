"""
EfficientNet-B0 with custom classifier head for binary classification.
"""

import torch
import torch.nn as nn
from torchvision import models
import config


def build_model(pretrained=True):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, config.NUM_CLASSES),
    )
    return model


def unfreeze_backbone(model, num_blocks_to_unfreeze=3):
    total_blocks = len(model.features)
    unfreeze_from = total_blocks - num_blocks_to_unfreeze
    for i, block in enumerate(model.features):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Unfroze last {num_blocks_to_unfreeze} blocks: {trainable:,}/{total:,} params trainable")


def get_optimizer(model, fine_tuning=False):
    if not fine_tuning:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return torch.optim.Adam([
            {"params": backbone_params, "lr": config.FINE_TUNE_LR},
            {"params": head_params, "lr": config.LEARNING_RATE},
        ], weight_decay=config.WEIGHT_DECAY)
