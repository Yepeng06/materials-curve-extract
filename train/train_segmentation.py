"""Train the curve segmentation U-Net on synthetic charts.

Usage (from the repo root, mci env):
  conda activate mci
  python scripts/gen_synthetic.py --out-dir data/train_synthetic --count 400 --seed 100
  python data/dataset_builder.py --out-dir data/train_platform --count 2000 --seed 20260815
  python train/train_segmentation.py --data-dir data/train_synthetic \
      --val-dir data/synthetic --epochs 40 --batch 16 \
      --out models/checkpoints/unet_curve.pt

The generator emits one 0/1 mask PNG per chart (<stem>_mask.png) whose only
foreground is the curve (drawn solid even for dashed curves), so the network
learns to ignore axes/grid/text and to complete dash gaps.

Resolution: pass --size to train at a different resolution (e.g. 512).  The
inference side must use the same resolution — set ``unet_size`` in
configs/baseline.yaml (default 256) to match, otherwise accuracy degrades.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mci.models.segmentation.unet import UNet, bce_dice_loss, dice_iou

SIZE = 256


def _list_pairs(data_dir: str):
    """Collect (image, mask) pairs; ``data_dir`` may be a comma-separated
    list of directories (e.g. gen_synthetic + platform-adapter data)."""
    pairs = []
    for d in data_dir.split(","):
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.png"))):
            if p.endswith("_mask.png"):
                continue
            mask = p[:-4] + "_mask.png"
            if os.path.exists(mask):
                pairs.append((p, mask))
    return pairs


class ChartDataset(Dataset):
    def __init__(self, pairs, augment: bool = False, size: int = SIZE):
        self.pairs = pairs
        self.augment = augment
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(img_path)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = _augment(img, mask)

        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        y = (torch.from_numpy(mask).float().unsqueeze(0) / 255.0 > 0.5).float()
        return x, y


def _augment(img: np.ndarray, mask: np.ndarray) -> tuple:
    """On-line augmentation (mirrors the generator's degradation palette)."""
    size = img.shape[0]
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
    if random.random() < 0.4:  # brightness/contrast
        g = random.uniform(0.8, 1.25)
        b = random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
    if random.random() < 0.3:  # mild rotation
        ang = random.uniform(-3, 3)
        m = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR,
                             borderValue=255)
        mask = cv2.warpAffine(mask, m, (size, size), flags=cv2.INTER_NEAREST,
                              borderValue=0)
    if random.random() < 0.25:  # JPEG noise
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,
                                             random.randint(60, 90)])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if random.random() < 0.3:  # random crop (chart region jitter)
        f = random.uniform(0.82, 1.0)
        cs = int(size * f)
        x0 = random.randint(0, size - cs)
        y0 = random.randint(0, size - cs)
        img = cv2.resize(img[y0:y0 + cs, x0:x0 + cs], (size, size),
                         interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask[y0:y0 + cs, x0:x0 + cs], (size, size),
                          interpolation=cv2.INTER_NEAREST)
    if random.random() < 0.2:  # mild perspective warp (scan skew)
        dx = random.uniform(2, 8)
        src = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
        dst = np.float32([[dx, 0], [size - dx, dx],
                          [0, size - dx], [size, size]])
        m = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, m, (size, size), flags=cv2.INTER_LINEAR,
                                  borderValue=255)
        mask = cv2.warpPerspective(mask, m, (size, size), flags=cv2.INTER_NEAREST,
                                   borderValue=0)
    if random.random() < 0.15:  # scan speckle / hairline noise
        m = random.random()
        img = img.copy()
        if m < 0.5:
            for _ in range(random.randint(1, 3)):  # thin dark/lit hairlines
                y0 = random.randint(0, size - 1)
                x0 = random.randint(0, size - 20)
                v = 0 if random.random() < 0.5 else 255
                cv2.line(img, (x0, y0), (x0 + random.randint(10, 60), y0), v, 1)
        else:
            noise = (np.random.default_rng().random(img.shape) < 0.0008)
            img[noise] = 0
            img[np.random.default_rng().random(img.shape) < 0.0008] = 255
    return img, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/train_synthetic")
    ap.add_argument("--val-dir", default="data/synthetic")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/checkpoints/unet_curve.pt")
    ap.add_argument("--size", type=int, default=SIZE,
                    help="training resolution (must match inference unet_size)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap training pairs (smoke tests / timing)")
    ap.add_argument("--amp", action="store_true", default=None,
                    help="enable mixed precision (auto-on for CUDA)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_amp = args.amp if args.amp is not None else device.startswith("cuda")
    print(f"device: {device}, amp: {use_amp}")

    pairs = _list_pairs(args.data_dir)
    val_pairs = _list_pairs(args.val_dir) if os.path.isdir(args.val_dir) else []
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"train pairs: {len(pairs)}, val pairs: {len(val_pairs)}")
    if len(pairs) < 10:
        print("ERROR: too few training images (did you run gen_synthetic with masks?)")
        return 1

    train_ds = ChartDataset(pairs, augment=True, size=args.size)
    val_ds = ChartDataset(val_pairs, augment=False, size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, base=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = bce_dice_loss(model(x), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot_loss += loss.item() * len(x)
            n += len(x)
        sched.step()

        model.eval()
        v_dice = v_iou = 0.0
        if len(val_loader):
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    prob = torch.sigmoid(model(x))
                    d, i = dice_iou(prob, y)
                    v_dice += d * len(x)
                    v_iou += i * len(x)
            v_dice /= len(val_ds)
            v_iou /= len(val_ds)
        else:
            v_dice = v_iou = float("nan")

        print(f"[{epoch:02d}/{args.epochs}] loss={tot_loss / max(n, 1):.4f} "
              f"val_dice={v_dice:.4f} val_iou={v_iou:.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.1f}s)")

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_iou": v_iou, "val_dice": v_dice}, args.out)
            print(f"  -> saved {args.out} (val_iou={v_iou:.4f})")

    print(f"done. best val_iou = {best_iou:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
