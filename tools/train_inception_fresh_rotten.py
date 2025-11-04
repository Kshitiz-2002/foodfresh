# tools/train_inception_fresh_rotten.py
import argparse, json, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def make_loaders(data_dir, img_size=299, bs=32, workers=4, binary=False):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    def class_map(name):
        if not binary: return name
        n = name.lower()
        return "fresh" if "fresh" in n else "rotten"

    def remap_dataset(root, tf):
        ds = datasets.ImageFolder(root=root, transform=tf)
        original = ds.samples
        remapped = []
        classes = sorted({class_map(Path(p).parent.name) for p,_ in original})
        class_to_idx = {c:i for i,c in enumerate(classes)}
        for p,_ in original:
            c = class_map(Path(p).parent.name)
            remapped.append((p, class_to_idx[c]))
        ds.samples = remapped
        ds.targets = [y for _,y in remapped]
        ds.classes = classes
        ds.class_to_idx = class_to_idx
        return ds

    train_root = str(Path(data_dir)/"train")
    val_root = str(Path(data_dir)/"val") if (Path(data_dir)/"val").exists() else str(Path(data_dir)/"test")

    train_ds = remap_dataset(train_root, train_tf)
    val_ds   = remap_dataset(val_root,   eval_tf)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def build_inception(num_classes, freeze_backbone=False):
    m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    if freeze_backbone:
        for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    if m.aux_logits:
        m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)
    return m

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_loaders(args.data, img_size=299, bs=args.batch_size, workers=args.workers, binary=args.binary)
    model = build_inception(num_classes=len(classes), freeze_backbone=args.freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    best_acc, best_path = 0.0, None

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if isinstance(out, tuple):  # inception returns (logits, aux)
                logits, aux = out
                loss = criterion(logits, y) + 0.4*criterion(aux, y)
            else:
                loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()*x.size(0)

        model.eval()
        correct = 0; total = 0; preds = []; gts = []; Va = 0.0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)[0] if isinstance(model(x), tuple) else model(x)
                p = logits.argmax(1)
                preds += p.cpu().tolist(); gts += y.cpu().tolist()
                correct += (p==y).sum().item(); total += y.size(0)
        Va = correct/total if total else 0.0

        print(f"Epoch {epoch+1}/{args.epochs} | val_acc={Va:.4f}")
        if Va > best_acc:
            best_acc = Va
            Path(args.out).mkdir(parents=True, exist_ok=True)
            best_path = str(Path(args.out)/"inception_fresh_rotten_best.pt")
            torch.save({"model":model.state_dict(),"classes":classes}, best_path)

    # final report
    if best_path:
        print("Best model:", best_path)
        rep = classification_report(gts, preds, target_names=classes, zero_division=0, output_dict=True)
        pd.DataFrame(rep).to_csv(str(Path(args.out)/"val_report.csv"))
        cm = confusion_matrix(gts, preds)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(str(Path(args.out)/"val_confusion_matrix.csv"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root containing train/ and val/ or test/")
    ap.add_argument("--out", default="models/classifiers/inception")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--binary", action="store_true", help="Map to 2 classes (fresh/rotten)")
    args = ap.parse_args()
    train(args)
