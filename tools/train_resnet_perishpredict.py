# # tools/train_resnet_perishpredict.py
# import argparse, json
# from pathlib import Path
# import torch, torch.nn as nn, torch.optim as optim
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from torchvision import datasets, transforms, models
# from sklearn.metrics import classification_report
# import numpy as np, pandas as pd

# def make_loaders(data_dir, img_size=224, bs=32, workers=4):
#     tf_train = transforms.Compose([
#         transforms.Resize(int(img_size*1.15)),
#         transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(0.2,0.2,0.2,0.1),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
#     ])
#     tf_eval = transforms.Compose([
#         transforms.Resize(int(img_size*1.1)),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
#     ])
#     train_ds = datasets.ImageFolder(root=str(Path(data_dir)/"train"), transform=tf_train)
#     val_root = Path(data_dir)/"val"
#     test_root = Path(data_dir)/"test"
#     val_ds = datasets.ImageFolder(root=str(val_root if val_root.exists() else test_root), transform=tf_eval)

#     # handle imbalance
#     class_counts = np.bincount(train_ds.targets)
#     class_weights = 1.0 / np.maximum(class_counts, 1)
#     sample_weights = class_weights[train_ds.targets]
#     sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

#     train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=workers, pin_memory=True)
#     val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
#     return train_loader, val_loader, train_ds.classes

# def build_resnet(num_classes=2, freeze=False):
#     m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
#     if freeze:
#         for p in m.parameters(): p.requires_grad = False
#     m.fc = nn.Linear(m.fc.in_features, num_classes)
#     return m

# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train_loader, val_loader, classes = make_loaders(args.data, bs=args.batch_size, workers=args.workers)
#     model = build_resnet(num_classes=len(classes), freeze=args.freeze).to(device)
#     criterion = nn.CrossEntropyLoss()
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

#     best_acc = 0.0; best_path=None
#     for epoch in range(args.epochs):
#         model.train()
#         for x,y in train_loader:
#             x,y = x.to(device), y.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(x), y)
#             loss.backward(); optimizer.step()
#         # val
#         model.eval(); correct=0; total=0; preds=[]; gts=[]
#         with torch.no_grad():
#             for x,y in val_loader:
#                 x,y = x.to(device), y.to(device)
#                 p = model(x).argmax(1)
#                 preds += p.cpu().tolist(); gts += y.cpu().tolist()
#                 correct += (p==y).sum().item(); total += y.size(0)
#         acc = correct/total if total else 0.0
#         print(f"Epoch {epoch+1}/{args.epochs} | val_acc={acc:.4f}")
#         if acc>best_acc:
#             best_acc=acc
#             Path(args.out).mkdir(parents=True, exist_ok=True)
#             best_path=str(Path(args.out)/"resnet50_perishpredict_best.pt")
#             torch.save({"model":model.state_dict(),"classes":classes}, best_path)

#     if best_path:
#         rep = classification_report(gts, preds, target_names=classes, zero_division=0, output_dict=True)
#         pd.DataFrame(rep).to_csv(str(Path(args.out)/"val_report.csv"))
#         print("Best model:", best_path)

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data", required=True)
#     ap.add_argument("--out", default="models/classifiers/resnet50")
#     ap.add_argument("--epochs", type=int, default=10)
#     ap.add_argument("--batch_size", type=int, default=32)
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--workers", type=int, default=4)
#     ap.add_argument("--freeze", action="store_true")
#     args = ap.parse_args()
#     train(args)


# tools/train_resnet_perishpredict.py
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def make_loaders(data_dir, img_size=224, bs=32, workers=4):
    tf_train = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(root=str(Path(data_dir)/"train"), transform=tf_train)
    val_root = Path(data_dir)/"val"
    test_root = Path(data_dir)/"test"
    val_ds = datasets.ImageFolder(root=str(val_root if val_root.exists() else test_root), transform=tf_eval)

    # class imbalance handling
    class_counts = np.bincount(train_ds.targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

def build_resnet(num_classes=2, freeze=False):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean()

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    v_loss_sum = 0.0; v_count = 0
    preds = []; gts = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        v_loss_sum += loss.item() * xb.size(0)
        v_count += xb.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(yb.cpu().tolist())
    v_loss = v_loss_sum / max(1, v_count)
    v_acc = (np.array(preds) == np.array(gts)).mean() if gts else 0.0
    return v_loss, v_acc, preds, gts

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_loaders(args.data, bs=args.batch_size, workers=args.workers)
    model = build_resnet(num_classes=len(classes), freeze=args.freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None

    writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0; best_path = None
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        tr_loss_sum = 0.0; tr_acc_sum = 0.0; tr_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                acc = accuracy_from_logits(logits, yb).item()

            bs = xb.size(0)
            tr_loss_sum += loss.item() * bs
            tr_acc_sum  += acc * bs
            tr_count    += bs
            global_step += 1

            avg_loss = tr_loss_sum / tr_count
            avg_acc  = tr_acc_sum  / tr_count
            cur_lr   = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}", lr=f"{cur_lr:.2e}")

            if writer:
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
                writer.add_scalar("train/batch_acc", acc, global_step)
                writer.add_scalar("train/lr", cur_lr, global_step)

        if scheduler is not None:
            scheduler.step()

        v_loss, v_acc, preds, gts = validate(model, val_loader, device, criterion)
        print(f"[Summary] epoch={epoch+1} tr_loss={(tr_loss_sum/max(1,tr_count)):.4f} tr_acc={(tr_acc_sum/max(1,tr_count)):.4f} val_loss={v_loss:.4f} val_acc={v_acc:.4f}")
        if writer:
            writer.add_scalar("val/epoch_loss", v_loss, epoch)
            writer.add_scalar("val/epoch_acc",  v_acc,  epoch)

        # per-epoch confusion matrix CSV
        cm = confusion_matrix(gts, preds, labels=list(range(len(classes))))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / f"val_confusion_matrix_epoch{epoch+1}.csv")

        # save best
        if v_acc > best_acc:
            best_acc = v_acc
            best_path = out_dir / "resnet50_perishpredict_best.pt"
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)

        # save last
        torch.save({"model": model.state_dict(), "classes": classes}, out_dir / "last_epoch.pt")

    # final report for last epoch predictions
    rep = classification_report(gts, preds, target_names=classes, zero_division=0, output_dict=True)
    pd.DataFrame(rep).to_csv(out_dir / "val_report_last_epoch.csv")
    if best_path:
        print("Best model:", best_path)

    if writer:
        writer.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root containing train/ and val/ or test/")
    ap.add_argument("--out", default="models/classifiers/resnet50", help="Output directory for checkpoints and CSVs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--freeze", action="store_true", help="Freeze backbone and train only classifier head")
    ap.add_argument("--cosine", action="store_true", help="Use CosineAnnealingLR over epochs")
    ap.add_argument("--log_dir", default="runs/perishpredict", help="TensorBoard log directory (set empty to disable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    args = ap.parse_args()
    train(args)
