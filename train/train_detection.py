"""Train a YOLOv8-nano structure detector (Phase B-4).

Builds an ultralytics dataset from the platform-style training set
(data/train_platform: <stem>.png + <stem>_yolo.txt, 6 classes) and
trains a detection model that will replace/augment the classical-CV
chart_structure detector (same ChartStructure output).

Usage (mci env):
    python train/train_detection.py --data-dir data/train_platform \
        --epochs 50 --batch 16 --imgsz 640 --model yolov8n.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

YOLO_NAMES = {
    0: "plot_area", 1: "x_axis_line", 2: "y_axis_line",
    3: "tick_label", 4: "legend_box", 5: "axis_title",
}


def build_dataset(data_dir: str, out_root: str, val_frac: float = 0.1,
                  limit: int = 0) -> str:
    """Copy images + yolo labels into an ultralytics layout; returns yaml path."""
    src = Path(data_dir)
    out = Path(out_root)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    pngs = sorted(p for p in src.glob("*.png") if not p.name.endswith("_mask.png"))
    if limit:
        pngs = pngs[:limit]
    n_val = max(1, int(len(pngs) * val_frac)) if len(pngs) > 10 else 0
    for i, p in enumerate(pngs):
        stem = p.stem
        yolo_txt = src / f"{stem}_yolo.txt"
        if not yolo_txt.exists():
            continue
        split = "val" if i >= len(pngs) - n_val else "train"
        shutil.copyfile(p, out / f"images/{split}/{p.name}")
        shutil.copyfile(yolo_txt, out / f"labels/{split}/{stem}.txt")

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        "# mci structure-detection dataset\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(YOLO_NAMES.items()))
        + "\n",
        encoding="utf-8",
    )
    n_tr = len(list((out / "images/train").glob("*.png")))
    n_va = len(list((out / "images/val").glob("*.png")))
    print(f"dataset ready: {n_tr} train / {n_va} val -> {yaml_path}")
    return str(yaml_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/train_platform")
    ap.add_argument("--out", default="data/detection")
    ap.add_argument("--limit", type=int, default=0,
                    help="train on the first N images only (quick smoke test)")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="mci_struct")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed in this env")
        return 1

    yaml_path = build_dataset(args.data_dir, args.out, limit=args.limit)
    model = YOLO(args.model)
    model.train(
        data=yaml_path, epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz, device=args.device, project=args.project,
        name=args.name, workers=2, patience=15,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
