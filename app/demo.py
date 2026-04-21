"""
app/demo.py
Gradio web demo for the Supermarket Shelf Monitor.

Usage:
    python app/demo.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from inference.detect import load_model, detect_products, find_empty_sections, draw_empty_gaps
from inference.inventory import process_all_gaps

# Load model once at startup
model = load_model()


def analyse_shelf(image: np.ndarray):
    """
    Main function called by Gradio.
    Takes a shelf image, runs detection, returns annotated image + report.
    """
    if image is None:
        return None, "Please upload a shelf image."

    # Save temp file (YOLO needs a file path)
    temp_path = "/tmp/shelf_input.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Run detection
    annotated_bgr, boxes, (h, w, _) = detect_products(model, temp_path)

    # Find empty sections
    empty_sections = find_empty_sections(boxes, w)

    # Draw red overlay on empty sections
    final_bgr = draw_empty_gaps(annotated_bgr, empty_sections)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)

    # Build report text
    lines = []
    lines.append(f"✅ Products detected: {len(boxes)}")
    lines.append(f"🔴 Empty sections: {len(empty_sections)}")
    lines.append("")

    if not empty_sections:
        lines.append("🎉 Shelf is fully stocked — no action needed!")
    else:
        lines.append("── Actions Required ──────────────────────")
        actions = process_all_gaps(empty_sections)
        for a in actions:
            lines.append(a["message"])

    report = "\n".join(lines)
    return final_rgb, report


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Supermarket Shelf Monitor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛒 Supermarket Shelf Monitor")
    gr.Markdown("Upload a shelf image to detect empty spots and generate restocking alerts.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Shelf Image", type="numpy")
            analyse_btn = gr.Button("Analyse Shelf", variant="primary")

        with gr.Column():
            output_image  = gr.Image(label="Detection Result")
            output_report = gr.Textbox(label="Restocking Report", lines=12)

    analyse_btn.click(
        fn=analyse_shelf,
        inputs=input_image,
        outputs=[output_image, output_report],
    )

    gr.Markdown("---")
    gr.Markdown("🔴 Red areas = empty shelf sections | 📦 = restock from warehouse | 🛒 = order from supplier")

if __name__ == "__main__":
    demo.launch(share=True)
