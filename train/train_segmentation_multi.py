"""Train the multi-curve instance-segmentation U-Net (Phase C).

Usage (from the repo root, mci env):
  python train/train_segmentation_multi.py       --data-dir data/train_platform,data/train_platform_single       --val-dir data/val_multi,data/val_single       --epochs 60 --batch 8 --size 512       --init models/checkpoints/unet_curve.pt       --out models/checkpoints/unet_multi_curve.pt

Instance masks: K=6 channels (max curves).  Per-instance channels are
REBUILT from the GT CSVs through the tick-label mapping (labels.json),
which is the exact basis the evaluator compares against (validated 0.4px
mean agreement with the single-curve 512 model).  The network learns
instance separation; the channel order is the curves.json order per image
(LineFormer-style instance regression).
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
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

SIZE = 512
K = 6  # max curves per chart (train_platform has 2-5)


def _list_pairs(data_dir: str):
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


def _fit_axis_from_labels(labels: list, axis: str, img_h: int):
    """Fit px<->value from GT label boxes (value = text, px = box centre).

    Matches the evaluation basis: GT CSV data points are compared
    against predictions through this same tick-label mapping, so a
    model trained on masks rebuilt through it extracts curves that
    agree with the GT interpolation (validated: 0.4px mean vs the
    single-curve 512 model on val_single).
    """
    from mci.utils import parse_number_text
    pts = []
    for it in labels:
        box = np.asarray(it["box"], dtype=float)
        c = box.mean(axis=0)
        v = parse_number_text(str(it["text"]))
        if v is None:
            continue
        if axis == "x" and c[1] > img_h - 80:
            pts.append((float(c[0]), v))
        elif axis == "y" and c[0] < 100 and c[1] < img_h - 85:
            pts.append((float(c[1]), v))
    if len(pts) < 2:
        return None
    pts.sort()
    p = np.array([q[0] for q in pts])
    v = np.array([q[1] for q in pts])
    if (v > 0).all() and (float(v.max()) / float(v.min()) > 100):
        a, b = np.polyfit(p, np.log10(v), 1)
        return ("log", float(a), float(b))
    a, b = np.polyfit(p, v, 1)
    return ("linear", float(a), float(b))


def _px_of(fit, v):
    kind, a, b = fit
    vv = np.asarray(v, dtype=np.float64)
    if kind == "log":
        vv = np.where(vv > 0, vv, 1e-9)  # non-positive values: clamp
        return (np.log10(vv) - b) / a
    return (vv - b) / a


def _meta_instance_masks(meta: dict, labels: list, curves_json: dict,
                        base_dir: str, img_size: tuple, k: int = K,
                        width: int = 3) -> np.ndarray:
    """K-channel instance masks rebuilt from the GT CSVs through the
    tick-label mapping (the evaluation basis).

    Replaces the curves_px polyline masks: those 16 sampled pixels do
    not trace the full rendered curve (160+ data points), so a model
    trained on them drifts from the GT interpolation.  Rebuilding the
    dense curve through the SAME mapping that the evaluator uses makes
    the training target and the acceptance metric consistent.
    """
    w, h = img_size
    out = np.zeros((k, h, w), np.uint8)
    if not curves_json or "curves" not in curves_json:
        return out
    fx = _fit_axis_from_labels(labels, "x", h)
    fy = _fit_axis_from_labels(labels, "y", h)
    if fx is None or fy is None:
        return out
    for i, c in enumerate(curves_json["curves"][:k]):
        csv_path = os.path.join(base_dir, c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        data = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        if len(data) < 2:
            continue
        gx = _px_of(fx, data[:, 0])
        gy = _px_of(fy, data[:, 1])
        pts = np.array([[int(round(x)), int(round(y))] for x, y in zip(gx, gy)],
                      dtype=np.int32)
        cv2.polylines(out[i], [pts], False, 255, thickness=width,
                      lineType=cv2.LINE_AA)
    return out




class ChartDataset(Dataset):
    def __init__(self, pairs, augment=False, size=SIZE, cache=None):
        self.pairs = pairs
        self.augment = augment
        self.size = size
        self.cache = cache  # dict path -> K-channel mask (for single-curve reuse)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sem = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or sem is None:
            raise FileNotFoundError(img_path)
        meta_p = img_path[:-4] + "_meta.json"
        meta = {}
        if os.path.exists(meta_p):
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
        key = img_path
        if self.cache is not None and key in self.cache:
            inst = self.cache[key]
        else:
            labels = []
            lab_p = img_path[:-4] + "_labels.json"
            if os.path.exists(lab_p):
                with open(lab_p, encoding="utf-8") as f:
                    labels = json.load(f)
            curves_json = {}
            cj_p = img_path[:-4] + "_curves.json"
            if os.path.exists(cj_p):
                with open(cj_p, encoding="utf-8") as f:
                    curves_json = json.load(f)
            inst = _meta_instance_masks(meta, labels, curves_json,
                                        os.path.dirname(img_path),
                                        (img.shape[1], img.shape[0]))
            if self.cache is not None:
                self.cache[key] = inst
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        inst = cv2.resize(inst.transpose(1, 2, 0), (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

        if self.augment:
            img, inst = _augment(img, inst)

        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        y = (torch.from_numpy(inst).float() / 255.0 > 0.5).float()
        return x, y


def _augment(img: np.ndarray, inst: np.ndarray) -> tuple:
    """Augment image + K-channel instance masks together (reuses baseline)."""
    size = img.shape[0]
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        inst = cv2.flip(inst, 2)
    if random.random() < 0.4:
        g = random.uniform(0.8, 1.25)
        b = random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        ang = random.uniform(-3, 3)
        m = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=255)
        for c in range(inst.shape[0]):
            inst[c] = cv2.warpAffine(inst[c], m, (size, size),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
    if random.random() < 0.25:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 90)])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if random.random() < 0.3:
        f = random.uniform(0.82, 1.0)
        cs = int(size * f)
        x0 = random.randint(0, size - cs)
        y0 = random.randint(0, size - cs)
        img = cv2.resize(img[y0:y0 + cs, x0:x0 + cs], (size, size), interpolation=cv2.INTER_LINEAR)
        inst_c = inst[:, y0:y0 + cs, x0:x0 + cs]
        inst = np.stack([cv2.resize(inst_c[c], (size, size),
                                    interpolation=cv2.INTER_NEAREST)
                         for c in range(inst.shape[0])])
    if random.random() < 0.15:
        noise = (np.random.default_rng().random(img.shape) < 0.0008)
        img = img.copy()
        img[noise] = 0
        img[np.random.default_rng().random(img.shape) < 0.0008] = 255
    return img, inst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/train_platform,data/train_platform_single")
    ap.add_argument("--val-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/checkpoints/unet_multi_curve.pt")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--init", default=None,
                    help="pretrained single-curve UNet (encoder weights shared)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-dir-limit", type=int, default=0,
                    help="cap training pairs PER directory (balanced subset)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_amp = device.startswith("cuda")
    print(f"device: {device}, amp: {use_amp}")

    pairs = _list_pairs(args.data_dir)
    val_pairs = _list_pairs(args.val_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if args.per_dir_limit > 0:
        per = []
        for p, m in pairs:
            d = os.path.dirname(p)
            cnt = sum(1 for q, _ in per if os.path.dirname(q) == d)
            if cnt < args.per_dir_limit:
                per.append((p, m))
        pairs = per
    print(f"train pairs: {len(pairs)}, val pairs: {len(val_pairs)}")
    if len(pairs) < 10:
        print("ERROR: too few training images")
        return 1

    cache: dict = {}
    train_ds = ChartDataset(pairs, augment=True, size=args.size, cache=cache)
    val_ds = ChartDataset(val_pairs, augment=False, size=args.size, cache=cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, base=64, out_channels=K).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        # encoder weights transfer; the output head (1 -> K channels) is new
        sd = {k: v for k, v in sd.items() if k.startswith("enc") or k.startswith("bottleneck")
              or k.startswith("up") or k.startswith("dec")}
        model.load_state_dict(sd, strict=False)
        print(f"initialized encoder from {args.init}")
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
                # channel weights: empty channels (target all-zero) are
                # down-weighted so the model does not learn to leave the
                # higher curve channels empty (4/5-curve charts are rarer
                # in the training distribution)
                w = (y.sum(dim=(2, 3)) > 0).float() * 0.7 + 0.3
                loss = (bce_dice_loss(model(x), y) * w).mean()
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
