# tools/clean_crops_keep_fruits.py
"""
Remove non-fruit crops from an existing crop folder based on filename class token.
Usage:
python .\tools\clean_crops_keep_fruits.py --crops .\crops --allowed apple banana orange
"""
import argparse
import os
from pathlib import Path

def clean(crops_dir, allowed):
    allowed_set = set([a.lower() for a in allowed])
    p = Path(crops_dir)
    if not p.exists():
        print("Crops dir not found:", crops_dir); return
    removed = 0; kept = 0
    for f in p.iterdir():
        if not f.is_file(): continue
        name = f.name.lower()
        # try to detect class token in filename (pattern: *_clsname_0.85.jpg)
        # We'll accept if any allowed token exists in filename.
        if any(tok in name for tok in allowed_set):
            kept += 1
        else:
            # remove file
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                print("Failed to remove", f, e)
    print(f"Removed {removed} files, kept {kept} files in {crops_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crops", required=True)
    p.add_argument("--allowed", nargs="*", default=["apple","banana","orange","grape","pear","pineapple","mango"])
    args = p.parse_args()
    clean(args.crops, args.allowed)
