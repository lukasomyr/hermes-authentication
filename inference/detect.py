"""
inference/detect.py
Runs YOLO detection on a shelf image, identifies empty gaps,
and triggers inventory logic.

Usage:
    python inference/detect.py --image path/to/shelf.jpg
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from inventory import process_all_gaps

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "runs/train/sku110k_finetune/weights/best.pt"
CONFIDENCE      = 0.3
NUM_SECTIONS    = 8   # how many shelf sections we expect left→right
# ──────────────────────────────────────────────────────────────────────────────


def load_model(weights: str = DEFAULT_WEIGHTS) -> YOLO:
    """Loads YOLO model. Falls back to pretrained if custom weights missing."""
    try:
        model = YOLO(weights)
        print(f"Loaded fine-tuned weights: {weights}")
    except Exception:
        print("Fine-tuned weights not found, using pretrained yolov8n.pt")
        model = YOLO("yolov8n.pt")
    return model


def detect_products(model: YOLO, image_path: str) -> tuple:
    """
    Runs YOLO on the image.
    Returns (annotated_image, list of bounding boxes).
    """
    results = model.predict(source=image_path, conf=CONFIDENCE, verbose=False)
    result  = results[0]

    image   = cv2.imread(image_path)
    boxes   = result.boxes.xyxy.cpu().numpy() if result.boxes else np.array([])
    annotated = result.plot()

    return annotated, boxes, image.shape


def find_empty_sections(boxes: np.ndarray, image_width: int, n_sections: int = NUM_SECTIONS) -> list:
    """
    Divides the image into N equal horizontal sections.
    A section is considered empty if no product bounding box center falls within it.
    Returns list of empty section names.
    """
    section_width = image_width / n_sections
    occupied = set()

    for box in boxes:
        x_center = (box[0] + box[2]) / 2
        section_idx = int(x_center / section_width)
        section_idx = min(section_idx, n_sections - 1)
        occupied.add(section_idx)

    section_names = [f"section_{chr(65 + i)}" for i in range(n_sections)]
    empty = [section_names[i] for i in range(n_sections) if i not in occupied]
    return empty


def draw_empty_gaps(image: np.ndarray, empty_sections: list, n_sections: int = NUM_SECTIONS) -> np.ndarray:
    """Draws red overlay on empty shelf sections."""
    h, w = image.shape[:2]
    section_width = w / n_sections
    overlay = image.copy()

    section_names = [f"section_{chr(65 + i)}" for i in range(n_sections)]
    for i, name in enumerate(section_names):
        if name in empty_sections:
            x1 = int(i * section_width)
            x2 = int((i + 1) * section_width)
            cv2.rectangle(overlay, (x1, 0), (x2, h), (0, 0, 255), -1)
            cv2.putText(overlay, "EMPTY", (x1 + 5, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return cv2.addWeighted(overlay, 0.4, image, 0.6, 0)


def run(image_path: str, weights: str = DEFAULT_WEIGHTS):
    model = load_model(weights)

    print(f"\nAnalysing: {image_path}")
    annotated, boxes, (h, w, _) = detect_products(model, image_path)

    print(f"Products detected: {len(boxes)}")

    empty_sections = find_empty_sections(boxes, w)
    print(f"Empty sections: {empty_sections if empty_sections else 'None – shelf is fully stocked!'}")

    final_image = draw_empty_gaps(annotated, empty_sections)
    output_path = image_path.replace(".", "_result.")
    cv2.imwrite(output_path, final_image)
    print(f"Result saved to: {output_path}")

    if empty_sections:
        print("\n── Inventory Actions ──────────────────────────────")
        actions = process_all_gaps(empty_sections)
        for a in actions:
            print(a["message"])

    return final_image, empty_sections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True, help="Path to shelf image")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to YOLO weights")
    args = parser.parse_args()
    run(args.image, args.weights)
