# FoodFreshAI

Local repo that provides:
- Fruit object detection (YOLOv8)
- Freshness classification (AlexNet, ResNet-50, Inception-v3)
- Expiry-date predictor using CNN features + optional metadata
- Dataset downloader (Kaggle), dataset manager, and inference API (FastAPI)

## Quick setup (Linux / macOS / WSL / Windows PowerShell)

1. Create python venv and install:
```bash
python -m venv venv
source venv/bin/activate      # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
