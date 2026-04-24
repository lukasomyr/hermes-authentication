"""
Single image inference.
"""

import argparse
import torch
from PIL import Image
from rembg import remove

import training.config as config
from training.dataset import get_eval_transforms
from training.model import build_model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    result = remove(img)
    white_bg = Image.new("RGBA", result.size, config.BACKGROUND_COLOR + (255,))
    white_bg.paste(result, mask=result.split()[3])
    final = white_bg.convert("RGB")
    transform = get_eval_transforms()
    return transform(final).unsqueeze(0)


@torch.no_grad()
def predict(model, image_tensor, device):
    model.eval()
    outputs = model(image_tensor.to(device))
    probs = torch.softmax(outputs, dim=1)
    confidence, idx = probs.max(1)
    return config.CLASS_NAMES[idx.item()], confidence.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=config.BEST_MODEL_PATH)
    args = parser.parse_args()

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=config.DEVICE))
    model = model.to(config.DEVICE)

    image_tensor = preprocess_image(args.image)
    pred, conf = predict(model, image_tensor, config.DEVICE)

    print(f"\n  Prediction: {pred.upper()}")
    print(f"  Confidence: {conf:.1%}")
    if pred.lower() == "fake":
        print("  Warning: This bag shows signs of being counterfeit.")
    else:
        print("  This bag appears authentic (screening only).")

if __name__ == "__main__":
    main()
