"""
Grad-CAM visualization: shows which regions of the bag the model focuses on.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

import config
from model import build_model
from inference import preprocess_image
from dataset import IMAGENET_MEAN, IMAGENET_STD


def generate_gradcam(model, image_tensor, target_class=None):
    model.eval()
    
    # Hook into the last conv layer of EfficientNet
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # Register hooks on the last feature block
    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    image_tensor = image_tensor.to(config.DEVICE)
    image_tensor.requires_grad_(True)
    output = model(image_tensor)
    probs = torch.softmax(output, dim=1)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass for the target class
    model.zero_grad()
    output[0, target_class].backward()
    
    # Compute Grad-CAM
    acts = activations[0].detach()
    grads = gradients[0].detach()
    weights = grads.mean(dim=[2, 3], keepdim=True)  # global avg pool of gradients
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)  # only positive contributions
    cam = F.interpolate(cam, size=(config.IMG_SIZE, config.IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize 0-1
    
    # Clean up hooks
    fh.remove()
    bh.remove()
    
    return cam, target_class, probs.detach().cpu().numpy()[0]


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu().squeeze() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def visualize_gradcam(image_path, save_path="outputs/gradcam_result.png"):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    
    image_tensor = preprocess_image(image_path)
    cam, pred_class, probs = generate_gradcam(model, image_tensor)
    original = denormalize(image_tensor)
    
    pred_label = config.CLASS_NAMES[pred_class]
    confidence = probs[pred_class]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Heatmap only
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(original)
    axes[2].imshow(cam, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Prediction: {pred_label.upper()} ({confidence:.1%})")
    axes[2].axis("off")
    
    plt.suptitle("Grad-CAM: Where is the model looking?", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    visualize_gradcam(args.image)
