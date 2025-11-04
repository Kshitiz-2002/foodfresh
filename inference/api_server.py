# inference/api_server.py
import os, io, json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# QR code
import qrcode  # pip install qrcode pillow

FRESH_TOKENS_DEFAULT  = ["fresh"]
ROTTEN_TOKENS_DEFAULT = ["rotten","spoiled","spoilt","bad","decay","decayed","mold","mould"]

# Configure via env vars or keep defaults
PT_WEIGHTS = os.getenv("PT_WEIGHTS", r".\models\classifiers\resnet50\resnet50_perishpredict_best.pt")
PT_ARCH    = os.getenv("PT_ARCH", "resnet")  # resnet|inception
KERAS_H5   = os.getenv("KERAS_H5", r".\models\inception_fresh_rotten_best.h5")
THRESHOLD  = float(os.getenv("THRESHOLD", "0.35"))

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def build_token_sets(classes, fresh_tokens, rotten_tokens):
    fresh_idx, rotten_idx = set(), set()
    lc = [c.lower() for c in classes]
    for i, cname in enumerate(lc):
        if any(t in cname for t in rotten_tokens): rotten_idx.add(i)
        if any(t in cname for t in fresh_tokens):  fresh_idx.add(i)
    if not fresh_idx and rotten_idx: fresh_idx = set(range(len(classes))) - rotten_idx
    if not rotten_idx and fresh_idx: rotten_idx = set(range(len(classes))) - fresh_idx
    return fresh_idx, rotten_idx

def aggregate_binary_probs(prob_vec: np.ndarray, fresh_idx, rotten_idx, keras_binary=False):
    if keras_binary:
        pR = float(prob_vec[0]); pf = 1.0 - pR
        return pf, pR
    pf = float(prob_vec[list(fresh_idx)].sum()) if fresh_idx else 0.0
    pR = float(prob_vec[list(rotten_idx)].sum()) if rotten_idx else 0.0
    tot = pf + pR
    if tot > 0: pf, pR = pf/tot, pR/tot
    return pf, pR

def load_pt(ckpt_path: str, arch: str, device: torch.device):
    p = Path(ckpt_path)
    if not p.exists():
        raise RuntimeError(f"PT weights not found: {ckpt_path}")
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    classes = ck["classes"]
    if arch == "inception":
        net = models.inception_v3(weights=None, aux_logits=False); resize, crop = 320, 299
    else:
        net = models.resnet50(weights=None); resize, crop = 256, 224
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(ck["model"]); net.eval().to(device)
    tfm = transforms.Compose([
        transforms.Resize(resize), transforms.CenterCrop(crop),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    fidx, ridx = build_token_sets(classes, FRESH_TOKENS_DEFAULT, ROTTEN_TOKENS_DEFAULT)
    return {"net": net, "classes": classes, "tfm": tfm, "fidx": fidx, "ridx": ridx}

def load_keras(h5_path: str):
    p = Path(h5_path)
    if not p.exists():
        raise RuntimeError(f"Keras .h5 not found: {h5_path}")
    m = load_model(str(p))
    ish = m.input_shape; ish = ish[0] if isinstance(ish, list) else ish
    h = ish[1] if len(ish)>=3 and ish[1] else 224
    w = ish[2] if len(ish)>=3 and ish[2] else 224
    def prep(pil: Image.Image):
        arr = np.array(pil.resize((w,h)).convert("RGB"), dtype=np.float32)/255.0
        return arr
    return {"model": m, "prep": prep}

def run_pt(state: Dict[str, Any], pil: Image.Image, device: torch.device):
    x = state["tfm"](pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(state["net"](x), dim=1)[0].cpu().numpy()
    classes = state["classes"]
    top_idx = int(np.argmax(probs)); top_lbl = classes[top_idx]; top_conf = float(probs[top_idx])
    pf, pR = aggregate_binary_probs(probs, state["fidx"], state["ridx"], keras_binary=False)
    bin_lbl = "rotten" if pR >= THRESHOLD else "fresh"; bin_conf = pR if bin_lbl=="rotten" else pf
    return {"pred": top_lbl, "conf": top_conf, "label_binary": bin_lbl, "prob_rotten": pR, "prob_fresh": pf}

def run_keras(state: Dict[str, Any], pil: Image.Image):
    x = state["prep"](pil)[None, ...]
    probs = state["model"].predict(x, verbose=0)
    if probs.ndim == 1:
        probs = probs[:, None]
    pj = probs[0]
    if probs.shape[1] == 1:
        pR = float(pj[0]); pf = 1.0 - pR
        label = "rotten" if pR >= THRESHOLD else "fresh"; conf = max(pR, pf)
        return {"pred": label, "conf": conf, "label_binary": label, "prob_rotten": pR, "prob_fresh": pf}
    else:
        top_idx = int(np.argmax(pj)); conf = float(pj[top_idx])
        return {"pred": str(top_idx), "conf": conf}

def _png_response(pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

def make_app() -> FastAPI:
    app = FastAPI(title="FoodFreshAI Classifier API")

    @app.on_event("startup")
    async def _load():
        app.state.device = device_auto()
        # Load PT (required)
        try:
            app.state.pt = load_pt(PT_WEIGHTS, PT_ARCH, app.state.device)
        except Exception as e:
            raise RuntimeError(f"PT load failed: {e}")
        # Load Keras (optional)
        try:
            app.state.k = load_keras(KERAS_H5)
        except Exception:
            app.state.k = None

    @app.get("/health")
    def health():
        return {"status": "ok", "pt_loaded": app.state.pt is not None, "keras_loaded": app.state.k is not None}

    @app.post("/v1/classify")
    async def classify(files: List[UploadFile] = File(...)):
        if app.state.pt is None and app.state.k is None:
            raise HTTPException(500, "No models loaded")
        results = []
        for uf in files:
            b = await uf.read()
            pil = bytes_to_pil(b)
            out = {}
            if app.state.pt is not None:
                out.update({f"pt_{k}": v for k,v in run_pt(app.state.pt, pil, app.state.device).items()})
            if app.state.k is not None:
                out.update({f"k_{k}": v for k,v in run_keras(app.state.k, pil).items()})
            ensR, ensF = [], []
            if "pt_prob_rotten" in out:
                ensR.append(out["pt_prob_rotten"]); ensF.append(out["pt_prob_fresh"])
            if "k_prob_rotten" in out:
                ensR.append(out["k_prob_rotten"]); ensF.append(out["k_prob_fresh"])
            if ensR:
                ensR = float(np.mean(ensR)); ensF = float(np.mean(ensF))
                ens_lbl = "rotten" if ensR >= THRESHOLD else "fresh"
                ens_conf = ensR if ens_lbl=="rotten" else ensF
                out.update({"ensemble_label": ens_lbl, "ensemble_prob_rotten": ensR, "ensemble_prob_fresh": ensF, "ensemble_conf": ens_conf})
            results.append({"file": uf.filename, **out})
        return JSONResponse(content={"results": results})

    # QR endpoints
    @app.get("/qr")
    def qr(url: Optional[str] = None, text: Optional[str] = None):
        target = url or text or "http://localhost:8000/docs"
        img = qrcode.make(target)
        return _png_response(img)

    @app.get("/qr/docs")
    def qr_docs(request: Request):
        base = str(request.base_url)
        img = qrcode.make(f"{base}docs")
        return _png_response(img)

    @app.get("/qr/classify")
    def qr_classify(request: Request):
        base = str(request.base_url)
        img = qrcode.make(f"{base}v1/classify")
        return _png_response(img)

    return app

app = make_app()
