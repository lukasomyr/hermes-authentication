"""
Hermes Authenticator - FastAPI backend
======================================

Serves the v2 web UI and exposes the authentication API.

Run:
    pip install -r requirements.txt
    # Place best_model.pth either at:
    #   ./best_model.pth
    #   ./Outputs/best_model.pth
    # or set HERMES_MODEL_PATH=/abs/path/to/best_model.pth
    uvicorn app:app --reload --port 8000

Then open:
    http://localhost:8000/         (v2 web UI)
    http://localhost:8000/docs     (auto-generated API docs)
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import config
from dataset import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from model import build_model

# Optional: rembg (background removal). The original inference.py uses it,
# but it's heavy and sometimes fails to install on certain platforms.
# We degrade gracefully: if rembg is missing, we skip BG removal.
try:
    from rembg import remove as _rembg_remove  # type: ignore
    HAS_REMBG = True
except Exception:  # pragma: no cover
    HAS_REMBG = False

# Optional: price estimator. If pricing/ isn't installed yet, the
# /api/estimate_price endpoint returns a clear 503 instead of crashing
# the whole server.
try:
    from pricing.price_data import estimate_price as _estimate_price  # type: ignore
    HAS_PRICING = True
    _PRICING_IMPORT_ERROR = ""
except Exception as _e:  # pragma: no cover
    HAS_PRICING = False
    _PRICING_IMPORT_ERROR = str(_e)


# ---------------------------------------------------------------------------
# Paths & device
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
STATIC_DIR = ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)


def _resolve_model_path() -> Path:
    """Find best_model.pth. Allow env override + a few sensible defaults."""
    env = os.getenv("HERMES_MODEL_PATH")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        ROOT / "best_model.pth",
        ROOT / "Outputs" / "best_model.pth",
        Path(config.BEST_MODEL_PATH),  # original hard-coded path
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find best_model.pth. Place it next to app.py (or in "
        "./Outputs/), or set HERMES_MODEL_PATH=/abs/path/to/best_model.pth."
    )


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Model loading (once, at startup)
# ---------------------------------------------------------------------------

_model: Optional[torch.nn.Module] = None
_eval_tf = get_eval_transforms()


def get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        path = _resolve_model_path()
        m = build_model(pretrained=False)
        state = torch.load(str(path), map_location=DEVICE)
        m.load_state_dict(state)
        m = m.to(DEVICE).eval()
        _model = m
        print(f"[hermes-auth] Loaded model from {path} on {DEVICE}")
    return _model


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def remove_background(pil_rgb_or_rgba: Image.Image) -> Image.Image:
    """rembg + white background fill, mirroring inference.preprocess_image."""
    if not HAS_REMBG:
        return pil_rgb_or_rgba.convert("RGB")
    img = pil_rgb_or_rgba.convert("RGBA")
    cut = _rembg_remove(img)
    white = Image.new("RGBA", cut.size, config.BACKGROUND_COLOR + (255,))
    white.paste(cut, mask=cut.split()[3])
    return white.convert("RGB")


def to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.detach().cpu().squeeze(0) * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Grad-CAM (clean reimplementation - returns numpy heatmap, no plt)
# ---------------------------------------------------------------------------

def compute_gradcam(model: torch.nn.Module,
                    image_tensor: torch.Tensor,
                    target_class: Optional[int] = None):
    """
    Returns (cam_norm[H,W] in [0,1], target_class:int, probs:np.ndarray[2]).

    Hooks the last EfficientNet feature block - same target layer the
    project's gradcam.py uses.
    """
    model.eval()
    activations, gradients = [], []

    def fwd_hook(_m, _i, output):
        activations.append(output)

    def bwd_hook(_m, _gi, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = image_tensor.to(DEVICE)
        x.requires_grad_(True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, target_class].backward()

        acts = activations[0].detach()
        grads = gradients[0].detach()
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(config.IMG_SIZE, config.IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        fh.remove()
        bh.remove()

    return cam, target_class, probs.detach().cpu().numpy()[0]


def jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Apply matplotlib's 'jet' colormap to a [0,1] heatmap -> RGB uint8."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    rgba = (cm.get_cmap("jet")(cam) * 255).astype(np.uint8)
    return rgba[..., :3]


def make_overlay(original_rgb01: np.ndarray, cam: np.ndarray, alpha=0.5) -> Image.Image:
    """Blend original (HxWx3 in [0,1]) with jet-colored cam."""
    base = (original_rgb01 * 255).astype(np.uint8)
    heat = jet_colormap(cam)
    blend = (base.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha)
    return Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))


def make_heatmap_image(cam: np.ndarray) -> Image.Image:
    return Image.fromarray(jet_colormap(cam))


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hermes Authenticator API",
    description="EfficientNet-B0 binary classifier for Hermes Birkin authenticity screening.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup():
    try:
        get_model()
    except FileNotFoundError as e:
        # Don't crash the server - /api/predict will return a clear error.
        print(f"[hermes-auth] WARNING: {e}")


@app.get("/api/health")
def health():
    info = {
        "status": "ok",
        "device": str(DEVICE),
        "rembg": HAS_REMBG,
        "classes": config.CLASS_NAMES,
        "img_size": config.IMG_SIZE,
        "model_loaded": _model is not None,
    }
    try:
        info["model_path"] = str(_resolve_model_path())
    except FileNotFoundError as e:
        info["model_path"] = None
        info["model_error"] = str(e)
    return info


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), gradcam: bool = True):
    """
    Run inference on a single image.

    Returns:
        {
          "verdict": "Real" | "Fake",
          "confidence": 0.943,
          "probs": {"Real": 0.943, "Fake": 0.057},
          "latency_ms": 412,
          "device": "mps",
          "preprocessed_image": "data:image/png;base64,...",
          "gradcam_overlay":   "data:image/png;base64,...",  # if gradcam=true
          "gradcam_heatmap":   "data:image/png;base64,..."   # if gradcam=true
        }
    """
    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    t0 = time.perf_counter()

    # Step 1+2: background removal -> white canvas -> RGB
    preprocessed = remove_background(pil)

    # Step 3: resize + normalize
    tensor = _eval_tf(preprocessed).unsqueeze(0)

    # Step 4+5: inference (+ optional Grad-CAM)
    if gradcam:
        cam, target_class, probs = compute_gradcam(model, tensor)
    else:
        with torch.no_grad():
            logits = model(tensor.to(DEVICE))
            probs_t = torch.softmax(logits, dim=1)
            confidence_t, idx_t = probs_t.max(dim=1)
            target_class = int(idx_t.item())
            probs = probs_t.detach().cpu().numpy()[0]
        cam = None

    latency_ms = int((time.perf_counter() - t0) * 1000)

    verdict = config.CLASS_NAMES[target_class]
    confidence = float(probs[target_class])

    payload = {
        "verdict": verdict,
        "confidence": confidence,
        "probs": {name: float(p) for name, p in zip(config.CLASS_NAMES, probs)},
        "latency_ms": latency_ms,
        "device": str(DEVICE),
        "preprocessed_image": to_data_url(preprocessed.resize(
            (config.IMG_SIZE, config.IMG_SIZE)
        )),
    }

    if cam is not None:
        original_rgb01 = denormalize(tensor)
        payload["gradcam_overlay"] = to_data_url(make_overlay(original_rgb01, cam))
        payload["gradcam_heatmap"] = to_data_url(make_heatmap_image(cam))

    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Price estimation
# ---------------------------------------------------------------------------

class PriceEstimateRequest(BaseModel):
    model: str
    size: str
    leather: str
    color: str
    hardware: Optional[str] = None
    condition: Optional[str] = None


@app.post("/api/estimate_price")
def estimate_price_endpoint(req: PriceEstimateRequest):
    """
    Returns a price range (low / mid / high in EUR) based on comparable
    Love Luxury listings, applying hardware + condition multipliers when an
    exact match is unavailable.

    Response shape (success):
        {
          "low": int, "mid": int, "high": int, "mean": int,
          "comparables": int,
          "multiplier": float,
          "notes": [str, ...]
        }

    Response shape (no comparables found):
        {"error": "...", "comparables": 0}
    """
    if not HAS_PRICING:
        raise HTTPException(
            status_code=503,
            detail=f"Pricing module not available: {_PRICING_IMPORT_ERROR}",
        )
    try:
        result = _estimate_price(
            model=req.model,
            size=req.size,
            leather=req.leather,
            color=req.color,
            hardware=req.hardware,
            condition=req.condition,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Price estimation failed: {e}")
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Static demo pages
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index_v2():
    return FileResponse(str(STATIC_DIR / "index_v2.html"))


@app.get("/v2")
def index_v2_alias():
    return FileResponse(str(STATIC_DIR / "index_v2.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
