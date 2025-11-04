# tools/train_days_regressor.py
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error

class DaysDataset(Dataset):
    def __init__(self, csv_path, img_root="", img_size=224, use_env=True, env_cols=("temp_c","rh")):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.use_env = use_env
        self.env_cols = list(env_cols)
        self.df["file_abs"] = self.df["file"].apply(lambda p: str((self.img_root/str(p)).resolve()))
        self.df = self.df[self.df["days_to_spoil"].notnull()]
        self.tf = transforms.Compose([
            transforms.Resize(int(img_size*1.1)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        # env normalization fit
        if self.use_env:
            env = self.df[self.env_cols].fillna(self.df[self.env_cols].median())
            self.env_mean = env.mean().values.astype(np.float32)
            self.env_std  = env.std(ddof=0).replace(0,1.0).values.astype(np.float32)
        else:
            self.env_mean = np.zeros(0, dtype=np.float32)
            self.env_std  = np.ones(0, dtype=np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["file_abs"]).convert("RGB")
        x = self.tf(img)
        y = np.float32(row["days_to_spoil"])
        if self.use_env:
            env_vals = row[self.env_cols].fillna(pd.Series(self.env_mean, index=self.env_cols)).values.astype(np.float32)
            env_norm = (env_vals - self.env_mean) / self.env_std
            e = torch.from_numpy(env_norm)
        else:
            e = torch.zeros(0)
        return x, e, y

class ImgEnvRegressor(nn.Module):
    def __init__(self, env_dim=0):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(feat + env_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    def forward(self, x, e=None):
        f = self.backbone(x)
        if e is not None and e.numel() > 0:
            if e.dim()==1: e = e.unsqueeze(0)
            f = torch.cat([f, e], dim=1)
        return self.head(f).squeeze(1)

def split_df(df, val_frac=0.2, seed=42):
    idx = np.arange(len(df)); rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(len(df)*(1-val_frac)); tr, va = idx[:cut], idx[cut:]
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = pd.read_csv(args.csv)
    tr_df, va_df = split_df(full, val_frac=args.val_frac, seed=args.seed)
    tr_csv, va_csv = Path(args.out)/"train_split.csv", Path(args.out)/"val_split.csv"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tr_df.to_csv(tr_csv, index=False); va_df.to_csv(va_csv, index=False)

    tr_ds = DaysDataset(tr_csv, img_root=args.img_root, img_size=args.imgsz, use_env=args.use_env, env_cols=args.env_cols)
    va_ds = DaysDataset(va_csv, img_root=args.img_root, img_size=args.imgsz, use_env=args.use_env, env_cols=args.env_cols)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    env_dim = len(tr_ds.env_mean) if args.use_env else 0
    model = ImgEnvRegressor(env_dim=env_dim).to(device)
    if args.freeze:
        for p in model.backbone.parameters(): p.requires_grad = False
    crit = nn.MSELoss()
    opt  = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    best_mae = 1e9; best_path=None
    for ep in range(args.epochs):
        model.train(); tr_loss=0; n=0
        for xb, eb, yb in tqdm(tr_loader, desc=f"Epoch {ep+1}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            eb = eb.to(device) if eb is not None and eb.numel()>0 else None
            opt.zero_grad(); pred = model(xb, eb); loss = crit(pred, yb); loss.backward(); opt.step()
            tr_loss += loss.item()*xb.size(0); n += xb.size(0)
        model.eval(); preds=[]; gts=[]
        with torch.no_grad():
            for xb, eb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                eb = eb.to(device) if eb is not None and eb.numel()>0 else None
                p = model(xb, eb); preds += p.cpu().tolist(); gts += yb.cpu().tolist()
        mae = mean_absolute_error(gts, preds); print(f"val_MAE={mae:.3f}")
        if mae < best_mae:
            best_mae = mae; best_path = Path(args.out)/"days_regressor_best.pt"
            torch.save({
                "model": model.state_dict(),
                "env_dim": env_dim,
                "imgsz": args.imgsz,
                "use_env": args.use_env,
                "env_cols": args.env_cols,
                "env_mean": tr_ds.env_mean.tolist(),
                "env_std": tr_ds.env_std.tolist(),
            }, best_path)
    print("Best:", best_path, "MAE:", best_mae)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: file, days_to_spoil, [temp_c,rh,...]")
    ap.add_argument("--img_root", default=".", help="Root to prefix file column")
    ap.add_argument("--out", default="models/regressors/days")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--use_env", action="store_true", help="Use environmental columns")
    ap.add_argument("--env_cols", nargs="*", default=["temp_c","rh"])
    args = ap.parse_args(); main(args)
