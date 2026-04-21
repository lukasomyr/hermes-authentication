# Supermarket Shelf Monitor

AI-powered shelf monitoring system using YOLO object detection to identify empty shelf gaps and trigger restocking alerts.

## Project Overview

This system detects products on supermarket shelves, identifies empty spots, checks a simulated inventory database, and automatically generates restocking tasks or purchase orders.

## Pipeline

```
Camera Image → YOLO Detection → Gap Analysis → Inventory Check → Restock Alert / Purchase Order
```

## Setup

```bash
git clone https://github.com/lukasomyr/supermarket-shelf-monitor.git
cd supermarket-shelf-monitor
pip install -r requirements.txt
```

## Dataset

We use the [SKU-110K dataset](https://github.com/eg4000/SKU110K_CVPR19):
- 11,762 supermarket shelf images
- 1.7M annotated bounding boxes
- Download and place in `data/SKU110K/`

## Usage

### Training
```bash
python training/train.py
```

### Inference (single image)
```bash
python inference/detect.py --image path/to/shelf.jpg
```

### Demo
```bash
python app/demo.py
```

## Project Structure

```
supermarket-shelf-monitor/
├── data/                   # Dataset (not committed)
│   └── SKU110K/
├── training/
│   └── train.py            # YOLO fine-tuning
├── inference/
│   ├── detect.py           # Detection + gap analysis
│   └── inventory.py        # Planogram + inventory logic
├── app/
│   └── demo.py             # Gradio demo
├── requirements.txt
└── README.md
```

## Team

Group project – IE University AI & Machine Learning course
