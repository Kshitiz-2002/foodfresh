# tools/crop_fruits_only.py
"""
Crop only fruit detections from images using a YOLO checkpoint.

Usage (PowerShell):
python .\tools\crop_fruits_only.py --weights .\models\weights\yolo11n.pt --images .\sample_images --out .\crops_fruits --conf 0.25 --allowed apple banana orange

Options:
 --allowed  list of class names to keep (space separated)
 --max_area_ratio  ignore boxes that cover > this fraction of the image (default 0.6)
 --min_area_ratio  ignore tiny boxes < this fraction (default 0.0005)
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

def crop_and_save(weights, images_dir, out_dir, conf=0.25, imgsz=640, allowed=None,
                  max_area_ratio=0.6, min_area_ratio=0.0005, pad=0.05):
    model = YOLO(weights)
    names = getattr(model.model, "names", None)
    print("Model classes mapping (id:name):")
    if names:
        print(names)
    else:
        print("No names metadata in model; use class ids in --allowed if needed.")

    allowed_set = set([a.lower() for a in (allowed or [])])
    if allowed_set:
        print("Allowed classes (case-insensitive):", allowed_set)
    else:
        print("No allowed classes specified: defaulting to all classes.")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    files = []
    p = Path(images_dir)
    if p.is_dir():
        files = [x for x in p.iterdir() if x.suffix.lower() in (".jpg",".jpeg",".png")]
    elif p.is_file():
        files = [p]
    else:
        raise RuntimeError("Images path not found: " + str(images_dir))

    stats = {"processed": 0, "saved": 0, "skipped_person": 0, "skipped_area": 0}
    for img_path in tqdm(files, desc="Running inference"):
        results = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, verbose=False)
        r = results[0]
        dets = list(r.boxes) if hasattr(r, "boxes") and r.boxes is not None else []
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        stats["processed"] += 1
        for i, d in enumerate(dets):
            cls_id = int(d.cls[0])
            conf_score = float(d.conf[0])
            # get class name if available
            cls_name = None
            if names and cls_id in names:
                cls_name = str(names[cls_id])
            else:
                cls_name = str(cls_id)

            # check allowed list (case-insensitive)
            if allowed_set and (cls_name.lower() not in allowed_set):
                # Not an allowed class -> skip
                # Count person skip separately if class is person
                if cls_name.lower() == "person":
                    stats["skipped_person"] += 1
                continue

            # bounding box coords
            x1, y1, x2, y2 = d.xyxy[0].tolist()
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            box_area = bw * bh
            img_area = w * h
            area_ratio = box_area / float(img_area)

            # area-based filtering (skip huge boxes likely persons or full-frame)
            if area_ratio > max_area_ratio or area_ratio < min_area_ratio:
                stats["skipped_area"] += 1
                continue

            # pad and crop
            dx = int(bw * pad)
            dy = int(bh * pad)
            xa = max(0, x1 - dx); ya = max(0, y1 - dy)
            xb = min(w, x2 + dx); yb = min(h, y2 + dy)
            crop = img[ya:yb, xa:xb]

            cls_safe = cls_name.replace(" ", "_")
            out_name = f"{img_path.stem}_det{i}_{cls_safe}_{conf_score:.2f}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, crop)
            stats["saved"] += 1

    print("Done. Stats:", stats)
    print("Crops saved to:", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--allowed", nargs="*", default=["apple","banana","orange","grape","pear","pineapple","mango"])
    p.add_argument("--max_area_ratio", type=float, default=0.6)
    p.add_argument("--min_area_ratio", type=float, default=0.0005)
    p.add_argument("--pad", type=float, default=0.05)
    args = p.parse_args()
    crop_and_save(args.weights, args.images, args.out, args.conf, args.imgsz, args.allowed,
                  args.max_area_ratio, args.min_area_ratio, args.pad)
