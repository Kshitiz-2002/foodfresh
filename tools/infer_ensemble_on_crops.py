# tools/infer_ensemble_on_crops.py
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# PyTorch
import torch
import torch.nn.functional as F
from torchvision import transforms, models

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

FRESH_TOKENS_DEFAULT  = ["fresh"]
ROTTEN_TOKENS_DEFAULT = ["rotten","spoiled","spoilt","bad","decay","decayed","mold","mould"]

def list_images(root, recursive=False, exts=(".jpg",".jpeg",".png",".bmp",".webp")):
    p = Path(root)
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    if not recursive:
        return [x for x in p.iterdir() if x.suffix.lower() in exts]
    return [x for x in p.rglob("*") if x.suffix.lower() in exts]

# ---------- PyTorch (.pt) ----------
def load_model_pt(ckpt_path, arch, device, half=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    if arch == "inception":
        net = models.inception_v3(weights=None, aux_logits=False)
        net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
        resize, crop = 320, 299
    else:
        net = models.resnet50(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
        resize, crop = 256, 224
    net.load_state_dict(ck["model"])
    net.eval().to(device)
    if half and device.type == "cuda":
        net.half()
    tfm = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return {"type":"pt","arch":arch,"classes":classes,"model":net,"tfm":tfm,"half":half}

def infer_batch_pt(state, device, pil_list):
    batch = []
    for img in pil_list:
        t = state["tfm"](img)
        if state["half"] and device.type == "cuda":
            t = t.half()
        batch.append(t)
    x = torch.stack(batch).to(device)
    with torch.no_grad():
        probs = F.softmax(state["model"](x), dim=1).cpu().numpy()
    return probs  # shape [B, C]

# ---------- Keras (.h5/.keras) ----------
def load_model_keras(h5_path):
    model = load_model(h5_path)
    # infer input size (H,W,3)
    ish = model.input_shape
    if isinstance(ish, list): ish = ish[0]
    h = ish[1] if len(ish)>=3 and ish[1] is not None else 224
    w = ish[2] if len(ish)>=3 and ish[2] is not None else 224
    # preprocessing: rescale 1/255 (common in Keras ImageDataGenerator)
    def prep(img_pil):
        arr = np.array(img_pil.resize((w,h)).convert("RGB"), dtype=np.float32)/255.0
        return arr
    return {"type":"keras","model":model,"prep":prep,"input_hw":(h,w)}

def infer_batch_keras(state, pil_list, batch_size=64):
    arrs = [state["prep"](img) for img in pil_list]
    x = np.stack(arrs, axis=0)
    probs = state["model"].predict(x, batch_size=batch_size, verbose=0)
    if probs.ndim == 1:
        probs = probs[:,None]
    return probs  # [B, C] (C=1 for binary sigmoid; otherwise softmax)

# ---------- Helpers ----------
def build_token_sets(classes, fresh_tokens, rotten_tokens):
    fresh_idx, rotten_idx = set(), set()
    lc = [c.lower() for c in classes]
    for i, cname in enumerate(lc):
        if any(tok in cname for tok in rotten_tokens): rotten_idx.add(i)
        if any(tok in cname for tok in fresh_tokens):  fresh_idx.add(i)
    if not fresh_idx and rotten_idx:
        fresh_idx = set(range(len(classes))) - rotten_idx
    if not rotten_idx and fresh_idx:
        rotten_idx = set(range(len(classes))) - fresh_idx
    return fresh_idx, rotten_idx

def aggregate_binary_probs(prob_vec, fresh_idx, rotten_idx, keras_binary=False):
    if keras_binary:  # single sigmoid column: p(rotten)=p[:,0]
        p_rotten = float(prob_vec[0])
        p_fresh  = 1.0 - p_rotten
        return p_fresh, p_rotten
    p_fresh = float(prob_vec[list(fresh_idx)].sum()) if fresh_idx else 0.0
    p_rotten = float(prob_vec[list(rotten_idx)].sum()) if rotten_idx else 0.0
    total = p_fresh + p_rotten
    if total > 0:
        p_fresh, p_rotten = p_fresh/total, p_rotten/total
    return p_fresh, p_rotten

def parse_models(model_specs):
    parsed = []
    for spec in model_specs:
        # pt:arch:path.pt   or   keras:path.h5
        if spec.startswith("pt:"):
            _, rest = spec.split("pt:", 1)
            if ":" not in rest:
                raise ValueError(f"PT spec needs arch:path.pt, got {spec}")
            arch, path = rest.split(":",1)
            parsed.append({"kind":"pt","arch":arch.strip(),"path":path.strip(),"key":Path(path).stem.replace('.','_')})
        elif spec.startswith("keras:"):
            _, path = spec.split("keras:",1)
            parsed.append({"kind":"keras","path":path.strip(),"key":Path(path).stem.replace('.','_')})
        else:
            raise ValueError(f"Unknown model spec: {spec}")
    return parsed

def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    files = list_images(args.images, recursive=args.recursive)
    if not files:
        print("No images found"); return

    fresh_tokens  = [t.lower() for t in (args.fresh_tokens or FRESH_TOKENS_DEFAULT)]
    rotten_tokens = [t.lower() for t in (args.rotten_tokens or ROTTEN_TOKENS_DEFAULT)]

    # Load models
    configs = []
    for spec in parse_models(args.models):
        if spec["kind"] == "pt":
            state = load_model_pt(spec["path"], spec["arch"], device, args.half)
            fidx, ridx = build_token_sets(state["classes"], fresh_tokens, rotten_tokens)
            configs.append({"key":spec["key"],"kind":"pt","state":state,"fresh_idx":fidx,"rotten_idx":ridx,"classes":state["classes"]})
        else:
            state = load_model_keras(spec["path"])
            # Keras class names unavailable from .h5 directly; accept user-provided class list file if needed
            # For binary h5, assume single sigmoid -> ['fresh','rotten'] mapping via threshold
            configs.append({"key":spec["key"],"kind":"keras","state":state,"fresh_idx":set(),"rotten_idx":set(),"classes":None})

    rows = []
    bs = max(1, args.batch)

    # Iterate in chunks for memory efficiency
    for i in range(0, len(files), bs):
        chunk_paths = files[i:i+bs]
        pil_list = [Image.open(p).convert("RGB") for p in chunk_paths]

        per_model_probs = {}
        per_model_meta  = {}

        for cfg in configs:
            key = cfg["key"]
            if cfg["kind"] == "pt":
                probs = infer_batch_pt(cfg["state"], device, pil_list)  # [B, C]
                per_model_probs[key] = probs
                per_model_meta[key]  = {"classes": cfg["classes"], "fresh_idx": cfg["fresh_idx"], "rotten_idx": cfg["rotten_idx"], "keras_binary": False}
            else:
                probs = infer_batch_keras(cfg["state"], pil_list, batch_size=bs)  # [B, C] (C=1 => sigmoid)
                per_model_probs[key] = probs
                # For Keras binary, derive binary by threshold on sigmoid later
                per_model_meta[key]  = {"classes": ["rotten"] if probs.shape[1]==1 else None, "fresh_idx": set(), "rotten_idx": set(), "keras_binary": (probs.shape[1]==1)}

        # Build records
        for j, pth in enumerate(chunk_paths):
            rec = {"file": str(pth)}
            ens_rotten_list = []
            ens_fresh_list  = []

            for key, probs in per_model_probs.items():
                meta = per_model_meta[key]
                pj = probs[j]
                if meta["keras_binary"]:
                    # pj shape [1], pj[0]=p(rotten)
                    p_fresh, p_rotten = aggregate_binary_probs(pj, set(), set(), keras_binary=True)
                    pred_lbl = "rotten" if p_rotten >= (args.threshold if args.threshold is not None else 0.5) else "fresh"
                    conf_lbl = max(p_rotten, p_fresh)
                else:
                    classes = meta["classes"]
                    top_idx = int(np.argmax(pj))
                    pred_lbl = classes[top_idx]
                    conf_lbl = float(pj[top_idx])
                    p_fresh, p_rotten = aggregate_binary_probs(pj, meta["fresh_idx"], meta["rotten_idx"], keras_binary=False)

                rec[f"pred_{key}"] = pred_lbl
                rec[f"conf_{key}"] = float(conf_lbl)
                rec[f"prob_rotten_{key}"] = float(p_rotten)
                rec[f"prob_fresh_{key}"]  = float(p_fresh)
                if args.save_probs and (not meta["keras_binary"]) and meta["classes"] is not None:
                    for ci, cname in enumerate(meta["classes"]):
                        rec[f"p_{key}_{cname}"] = float(pj[ci])

                ens_rotten_list.append(float(p_rotten))
                ens_fresh_list.append(float(p_fresh))

            ens_rotten = float(np.mean(ens_rotten_list)) if ens_rotten_list else 0.0
            ens_fresh  = float(np.mean(ens_fresh_list))  if ens_fresh_list  else 0.0
            if args.threshold is not None:
                ens_label = "rotten" if ens_rotten >= args.threshold else "fresh"
                ens_conf  = ens_rotten if ens_label=="rotten" else ens_fresh
            else:
                ens_label = "rotten" if ens_rotten >= ens_fresh else "fresh"
                ens_conf  = max(ens_rotten, ens_fresh)
            rec.update({"ensemble_label": ens_label, "ensemble_prob_rotten": ens_rotten, "ensemble_prob_fresh": ens_fresh, "ensemble_conf": ens_conf})
            rows.append(rec)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if args.move:
        for r in rows:
            dst = Path(args.move)/r["ensemble_label"]
            dst.mkdir(parents=True, exist_ok=True)
            Path(r["file"]).replace(dst/Path(r["file"]).name)

if __name__ == "__main__":
    import numpy as np
    ap = argparse.ArgumentParser(description="Ensemble inference with PyTorch .pt and Keras .h5 models")
    ap.add_argument("--models", nargs="+", required=True, help="Use 'pt:arch:path.pt' and/or 'keras:path.h5' (e.g., pt:resnet:...pt keras:...h5)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/preds_ensemble.csv")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=None, help="Binary decision threshold on ensemble rotten probability")
    ap.add_argument("--save_probs", action="store_true", help="Save per-class probs for PT models")
    ap.add_argument("--fresh_tokens", nargs="*", default=None)
    ap.add_argument("--rotten_tokens", nargs="*", default=None)
    ap.add_argument("--move", default=None, help="Move images by ensemble_label")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")
    ap.add_argument("--half", action="store_true", help="Half precision for PT models on CUDA")
    args = ap.parse_args()
    main(args)
