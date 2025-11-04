# tools/crop_detections_for_classifier.py
import argparse, os
from ultralytics import YOLO
from pathlib import Path
import cv2

def crop_and_save(weights, images_dir, out_dir, conf=0.25, imgsz=640, pad=0.05):
    model = YOLO(weights)
    names = getattr(model.model, "names", None)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in Path(images_dir).iterdir():
        if p.suffix.lower() not in (".jpg",".png",".jpeg"): continue
        results = model.predict(source=str(p), imgsz=imgsz, conf=conf, verbose=False)
        r = results[0]
        dets = list(r.boxes) if hasattr(r, "boxes") and r.boxes is not None else []
        img = cv2.imread(str(p))
        h,w = img.shape[:2]
        for i, d in enumerate(dets):
            x1,y1,x2,y2 = d.xyxy[0].tolist()
            # add small pad
            dx = (x2-x1) * pad
            dy = (y2-y1) * pad
            xa = max(0, int(x1-dx)); ya = max(0, int(y1-dy))
            xb = min(w, int(x2+dx)); yb = min(h, int(y2+dy))
            crop = img[ya:yb, xa:xb]
            clsid = int(d.cls[0])
            clsname = names[clsid] if names else str(clsid)
            out_name = f"{p.stem}_det{i}_{clsname}_{float(d.conf[0]):.2f}.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), crop)
    print("Crops saved to", out_dir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    crop_and_save(args.weights, args.images, args.out, args.conf, args.imgsz)
