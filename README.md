# Hermès-Auth

AI-powered authentication and price estimation for Hermès Birkin bags. Upload a photo — get an instant verdict, a Grad-CAM heatmap showing where the model looked, and a data-driven price estimate.

Built with EfficientNet-B0, FastAPI, and 687 hand-curated images.

---

## Demo Setup

### 1. Clone the repo
```bash
git clone https://github.com/lukasomyr/hermes-authentication.git
cd hermes-authentication
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the trained model
Download `best_model.pth` from [Google Drive](https://drive.google.com/file/d/1dkEI1XgoDN41-h8XoHlCxZIqIYe6fOn1/view?usp=sharing) and place it in the root folder:
```
hermes-authentication/
└── best_model.pth
```

### 4. Run the demo
```bash
uvicorn app:app --reload --port 8000
```
Then open `http://localhost:8000` in your browser.

---

## Project Structure

```
hermes-authentication/
├── app.py                  # FastAPI backend + demo server
├── config.py               # Model config & hyperparameters
├── model.py                # EfficientNet-B0 architecture
├── dataset.py              # Dataset & transforms
├── requirements.txt
├── static/
│   └── index_v2.html       # Demo UI
├── pricing/
│   ├── price_data.py       # Lookup + multiplier logic
│   ├── price_database.csv  # Scraped Birkin listings
│   ├── price_estimator.py  # CLI interface
│   └── price_fetcher.py    # Selenium scraper
├── training/
│   ├── train.py            # Two-phase training with early stopping
│   ├── cross_validate.py   # 3-fold cross-validation
│   ├── evaluate.py         # Test set evaluation + confusion matrix
│   ├── preprocess.py       # Background removal + train/val/test split
│   ├── gradcam.py          # Grad-CAM visualization
│   ├── inference.py        # Single image inference
│   └── mine_hard_negatives.py  # Playwright scraper for hard-negative mining
└── notebooks/
    └── hermes-authenticator-bags.ipynb
```

---

## Model

**Architecture:** EfficientNet-B0 pretrained on ImageNet, fine-tuned on 687 curated Hermès images.

**Training strategy:**
- Phase 1 (epochs 1–5): classifier head only, backbone frozen
- Phase 2 (epoch 6–30): last 3 backbone blocks unfrozen, differential learning rate (1e-5 backbone / 1e-3 head)
- 81.7% of parameters fine-tuned

**Results:**

| Metric | Value |
|---|---|
| Test Accuracy | 95.65% |
| Fake Recall | 96.67% |
| 3-Fold CV Accuracy | 97.04% ± 1.7% |
| False Positive Rate (post hard-negative mining) | 12.0% |

**3-Tier confidence system:**
- > 80% Real → Likely Authentic
- 50–80% → Uncertain — recommend expert review
- > 80% Fake → Likely Counterfeit

---

## Dataset

All 687 images were self-collected — no pre-existing dataset used.

**Authentic (490 images):**
- Scraped from The RealReal, Rebag, and Vestiaire Collective
- All listings pre-authenticated by in-house experts
- Hard-negative mining: custom Playwright scraper identifies false positives on TheRealReal and adds them to training — reduced false positive rate from 78% to 12%

**Counterfeit (197 images):**
- Sourced from Reddit (r/RepLadies), DHGate, YouTube comparisons
- Includes superfake catalog from a replica manufacturer — the hardest possible classification challenge

**Preprocessing:**
```bash
python training/preprocess.py
```
Removes backgrounds (rembg + BiRefNet), resizes to 224×224, splits 70/15/15.

---

## Training

```bash
python training/preprocess.py      # Step 1: preprocess + split
python training/train.py           # Step 2: train
python training/evaluate.py        # Step 3: evaluate
python training/cross_validate.py  # Optional: 3-fold CV
```

---

## Hard-Negative Mining

```bash
python training/mine_hard_negatives.py
```

Scrapes TheRealReal Birkin listings, runs them through the model, and saves false positives as hard negatives for retraining.

---

## Price Estimation

Rule-based lookup against a database of real resale listings. No ML — deliberately chosen because a regression model would overfit on the available data.

```bash
python pricing/price_estimator.py --model Birkin --size 30 --leather Togo --color Noir --condition "Pre-Loved"
```

---

## Disclaimer

This tool is intended for educational and research purposes only. Authentication results should not be used as the sole basis for purchase decisions. Always consult a certified authentication service for high-value items.

---

## Course

IE University — AI: Machine Learning & Analytics  
Final Project — Aisha Kapur, Danielle M. Lois, Matteo Benjamin, Lukas Ohmayer, Guilherme Teixeira, Ignacio Uriz