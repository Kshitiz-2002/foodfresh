# tools/infer_keras_shelf_life.py
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model  # Keras load_model API [web:252]

# Default subclasses from the repo: Fresh, Semifresh, SemiRotten, Rotten [attached_file:1]
DEFAULT_CLASSES = ["Fresh","Semifresh","SemiRotten","Rotten"]

def list_images(root, recursive=False, exts=(".jpg",".jpeg",".png",".bmp",".webp")):
    p = Path(root)
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    it = p.rglob("*") if recursive else p.iterdir()
    return [x for x in it if x.suffix.lower() in exts]

def load_keras_model(path):
    m = load_model(path)  # Keras model deserialization [web:252]
    ish = m.input_shape
    if isinstance(ish, list): ish = ish[0]
    h = ish[1] if len(ish)>=3 and ish[1] else 224
    w = ish[2] if len(ish)>=3 and ish[2] else 224
    return m, (h, w)

def preprocess_batch(pils, target_hw):
    h, w = target_hw
    arrs = []
    for im in pils:
        arr = np.array(im.resize((w,h)).convert("RGB"), dtype=np.float32) / 255.0  # ImageDataGenerator convention [web:262]
        arrs.append(arr)
    return np.stack(arrs, axis=0)

def map_binary(label_text):
    lt = label_text.lower()
    if "rotten" in lt or "semirotten" in lt or "spoiled" in lt or "spoilt" in lt:  # collapse to rotten [attached_file:1]
        return "rotten"
    return "fresh"

def main(args):
    # Load models
    cat_model, cat_hw = load_keras_model(args.category)  # category.h5 [attached_file:1][web:252]
    days_model, days_hw = load_keras_model(args.days)    # full_model.h5 [attached_file:1][web:252]

    files = list_images(args.images, recursive=args.recursive)
    if not files:
        print("No images found"); return

    # Class names: can be overridden from CLI if repo’s class order differs [attached_file:1]
    classes = args.classes if args.classes else DEFAULT_CLASSES

    rows = []
    bs = max(1, args.batch)
    for i in range(0, len(files), bs):
        chunk = files[i:i+bs]
        pils = [Image.open(p).convert("RGB") for p in chunk]
        x_cat  = preprocess_batch(pils, cat_hw)   # 1/255 scaled [web:262]
        x_days = preprocess_batch(pils, days_hw)  # 1/255 scaled [web:262]

        # Predict category (softmax)
        p_cat = cat_model.predict(x_cat, batch_size=bs, verbose=0)  # Keras predict [web:252]
        # Predict shelf-life days (regression)
        y_days = days_model.predict(x_days, batch_size=bs, verbose=0).reshape(-1)  # Keras predict [web:252]

        for j, pth in enumerate(chunk):
            probs = p_cat[j]
            # Handle binary/single-node edge case
            if probs.ndim == 0 or probs.shape == ():
                # unlikely for category.h5, but guard regardless [attached_file:1]
                pred_idx, pred_conf = 0, float(probs)
            else:
                pred_idx = int(np.argmax(probs))
                pred_conf = float(probs[pred_idx])

            pred_text = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            label_binary = map_binary(pred_text)  # Fresh/Semifresh -> fresh; SemiRotten/Rotten -> rotten [attached_file:1]
            days_pred = float(np.clip(y_days[j], 0.0, args.max_days))  # clamp range for stability [attached_file:1]

            row = {
                "file": str(pth),
                "category_pred": pred_text,
                "category_conf": pred_conf,
                "label_binary": label_binary,
                "days_pred": days_pred,
            }
            if args.save_probs and probs.ndim > 0:
                for k, cname in enumerate(classes):
                    if k < probs.shape[0]:
                        row[f"p_{cname}"] = float(probs[k])
            rows.append(row)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outp, index=False)
    print(f"Saved: {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help="Path to category.h5 (Multi-class freshness model)")  # repo file [attached_file:1]
    ap.add_argument("--days",     required=True, help="Path to full_model.h5 (Shelf-life regressor)")       # repo file [attached_file:1]
    ap.add_argument("--images",   required=True, help="Folder or image path")
    ap.add_argument("--out",      default="outputs/keras_shelf_life.csv")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--batch",    type=int, default=64)
    ap.add_argument("--max_days", type=float, default=30.0)
    ap.add_argument("--classes",  nargs="*", default=None, help="Override category class list if order differs from model")
    ap.add_argument("--save_probs", action="store_true")
    args = ap.parse_args()
    main(args)
