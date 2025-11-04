# tools/aggregate_yolo_stats.py
import argparse, os
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from tqdm import tqdm

def aggregate(weights, images_dir, conf=0.25, imgsz=640):
    model = YOLO(weights)
    names = getattr(model.model, "names", None)
    files = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")]
    stats = {"images": 0, "detections": 0, "per_class_counts": {}, "confidences": []}
    for f in tqdm(files):
        results = model.predict(source=str(f), imgsz=imgsz, conf=conf, verbose=False)
        r = results[0]
        dets = list(r.boxes) if hasattr(r, "boxes") and r.boxes is not None else []
        stats["images"] += 1
        stats["detections"] += len(dets)
        for d in dets:
            cid = int(d.cls[0])
            stats["per_class_counts"].setdefault(cid, 0)
            stats["per_class_counts"][cid] += 1
            stats["confidences"].append(float(d.conf[0]))
    print("Images processed:", stats["images"])
    print("Total detections:", stats["detections"])
    if names:
        print("Per-class counts (name -> count):")
        for cid, c in stats["per_class_counts"].items():
            print(f"  {names[cid]} -> {c}")
    else:
        print("Per-class counts (id -> count):", stats["per_class_counts"])
    if stats["confidences"]:
        arr = np.array(stats["confidences"])
        print(f"Mean confidence: {arr.mean():.3f}, median: {np.median(arr):.3f}, min: {arr.min():.3f}, max: {arr.max():.3f}")
    else:
        print("No detections with conf>=", conf)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    aggregate(args.weights, args.images, args.conf, args.imgsz)
