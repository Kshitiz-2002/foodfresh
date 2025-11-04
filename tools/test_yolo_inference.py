# tools/test_yolo_inference.py
"""
Run inference with a YOLOv8 .pt checkpoint on one or many images,
print model classes, save annotated outputs, and print basic stats.

Usage (PowerShell):
PS> python .\tools\test_yolo_inference.py --weights .\models\weights\yolo11n.pt --images .\sample_images --out .\outputs --conf 0.25

You can pass a single image file or a folder of images.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm

def annotate_and_save(image_bgr, detections, names, out_path):
    img = image_bgr.copy()
    for det in detections:
        x1,y1,x2,y2 = map(int, det.xyxy[0].tolist())
        conf = float(det.conf[0])
        cls_id = int(det.cls[0])
        label = f"{names[cls_id] if names else cls_id}: {conf:.2f}"
        # box
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        # text background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), (0,255,0), -1)
        cv2.putText(img, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)

def run_on_path(weights, images_path, out_dir, conf_thresh=0.25, imgsz=640, device=None):
    # load
    print("Loading model:", weights)
    model = YOLO(weights)  # ultralytics handles device automatically (cuda if available)
    # show classes
    names = getattr(model.model, "names", None)
    if names:
        print("Model classes:", names)
    else:
        print("Model has no names attribute; class ids will be integers.")

    # accept file or folder
    p = Path(images_path)
    if p.is_dir():
        files = [x for x in p.iterdir() if x.suffix.lower() in [".jpg",".jpeg",".png"]]
    elif p.is_file():
        files = [p]
    else:
        raise RuntimeError("Images path not found: " + str(images_path))

    os.makedirs(out_dir, exist_ok=True)
    stats = {"images": 0, "dets": 0}
    for f in tqdm(files, desc="Running inference"):
        results = model.predict(source=str(f), imgsz=imgsz, conf=conf_thresh, verbose=False)
        r = results[0]
        dets = []
        # ultralytics result.boxes is a Boxes object; iterate
        if hasattr(r, "boxes") and r.boxes is not None:
            dets = list(r.boxes)
        img_bgr = cv2.imread(str(f))
        out_path = Path(out_dir) / f.name
        annotate_and_save(img_bgr, dets, names, out_path)
        stats["images"] += 1
        stats["dets"] += len(dets)

    print(f"Saved annotated images to: {out_dir}")
    print(f"Processed {stats['images']} images, total detections: {stats['dets']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to yolo .pt weights")
    p.add_argument("--images", required=True, help="Single image or folder")
    p.add_argument("--out", default="outputs", help="Output folder for annotated images")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    run_on_path(args.weights, args.images, args.out, args.conf, args.imgsz)
