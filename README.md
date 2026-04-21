# Hermès Bag Authenticator

A deep learning system that authenticates Hermès Birkin bags from images using EfficientNet-B0. Upload a photo of a bag and the model predicts whether it is **authentic or fake**, with confidence score and Grad-CAM visualization showing which parts of the bag influenced the decision.

---

## Project Structure

```
hermes-authentication/
├── data/
│   ├── birkin_prices.csv       # Filename → price mapping (scraped from Love Luxury)
│   └── price_fetcher.py        # Script to automatically fetch prices from Love Luxury
│
├── training/
│   ├── config.py               # Central config (paths, hyperparameters, device)
│   ├── dataset.py              # PyTorch Dataset + augmentation transforms
│   ├── model.py                # EfficientNet-B0 with custom classifier head
│   ├── train.py                # Two-phase training with early stopping
│   ├── evaluate.py             # Test set evaluation, confusion matrix, error analysis
│   ├── gradcam.py              # Grad-CAM visualization
│   ├── inference.py            # Single image inference
│   └── preprocess.py           # Background removal + train/val/test split
│
├── inference/                  # Inference scripts for deployment
├── app/                        # Gradio demo (coming soon)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/hermes-authentication.git
cd hermes-authentication
pip install -r requirements.txt
```

---

## Data

### Authentic Images
Scraped from [Love Luxury](https://loveluxury.co.uk/shop/birkin/) — high quality product photos with white background.

### Fake Images
Collected from replica review communities — matched for image quality and background consistency.

### Preprocessing
Background removal is handled automatically using `rembg`:

```bash
python training/preprocess.py
```

This removes backgrounds, resizes images to 224×224, and splits the dataset into train (70%) / val (15%) / test (15%).

**Expected folder structure before preprocessing:**
```
data/raw/
├── Real/
│   ├── image1.jpg
│   └── ...
└── Fake/
    ├── image1.jpg
    └── ...
```

---

## Training

Update `RAW_DATA_DIR` in `training/config.py` to point to your dataset, then run:

```bash
python training/preprocess.py   # Step 1: preprocess + split
python training/train.py        # Step 2: train model
python training/evaluate.py     # Step 3: evaluate on test set
```

### Training Strategy
- **Phase 1** (epochs 1–5): Only the classifier head is trained, backbone frozen
- **Phase 2** (epoch 6+): Last 3 blocks of EfficientNet unfrozen with lower LR
- Early stopping with patience 7

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Image size | 224×224 |
| Batch size | 32 |
| Initial LR | 1e-3 |
| Fine-tune LR | 1e-5 |
| Epochs | 30 (early stopping) |
| Optimizer | Adam |

---

## Inference

```bash
python training/inference.py --image path/to/bag.jpg
```

Output:
```
  Prediction: REAL
  Confidence: 94.2%
  This bag appears authentic (screening only).
```

---

## Grad-CAM Visualization

```bash
python training/gradcam.py --image path/to/bag.jpg
```

Generates a heatmap showing which regions of the bag the model focused on when making its prediction.

---

## Price Fetcher

To fetch prices for your image dataset automatically:

```bash
pip install selenium webdriver-manager pandas
```

Edit `IMAGE_FOLDER` in `data/price_fetcher.py`, then:

```bash
python data/price_fetcher.py
```

Output: `birkin_prices.csv` with columns `filename | price_gbp`.

---

## Requirements

```
torch
torchvision
efficientnet-pytorch
rembg
pillow
scikit-learn
matplotlib
seaborn
pandas
tqdm
gradio
selenium
webdriver-manager
```

---

## Disclaimer

This tool is intended for educational and research purposes only. Authentication results should not be used as the sole basis for purchase decisions. Always consult a certified authentication service for high-value items.

---

## Course

IE University – AI: Machine Learning & Analytics
Final Project – Group Submission