"""S2: prepare 2x Lanczos-upsampled copies of a chart image set.

For every PNG (excluding *_mask.png) in --src-dir:
  * image  -> cv2.resize(..., INTER_LANCZOS4) x2, written to --out-dir
  * GT csv / *_meta.json / *_curves.json -> copied unchanged (data-domain GT
    is scale invariant)
  * *_labels.json -> copied with every OCR box scaled x2 (stub OCR boxes must
    live in the upsampled image's coordinate space)
  * *_mcg.json  -> copied with pixel_points / curves_px scaled x2 (keeps the
    exact fractional GT pixels usable for pixel-domain bias analysis)

--images may name a file containing one stem (basename without extension)
per line to select a subset; otherwise every image is processed.

Usage:
  python scripts/diag_subpixel_s2_prep.py \
      --src-dir data/eval_phased_500_single --out-dir data/experiments_s1s2/single_x2
  python scripts/diag_subpixel_s2_prep.py \
      --src-dir data/val_multi --out-dir data/experiments_s1s2/multi_x2 \
      --images data/experiments_s1s2/multi_select.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

SCALE = 2.0


def scale_boxes(labels_path: str, out_path: str) -> int:
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "box" in item:
            item["box"] = [[c * SCALE for c in corner] for corner in item["box"]]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return len(data)


def scale_mcg(mcg_path: str, out_path: str) -> None:
    with open(mcg_path, "r", encoding="utf-8") as f:
        mcg = json.load(f)
    for c in mcg.get("curves", []):
        if "pixel_points" in c:
            c["pixel_points"] = [[p[0] * SCALE, p[1] * SCALE] for p in c["pixel_points"]]
        if "bbox_xyxy" in c:
            c["bbox_xyxy"] = [v * SCALE for v in c["bbox_xyxy"]]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mcg, f, ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--images", default=None,
                    help="file with selected stems (basename without ext), one per line")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    selected = None
    if args.images and os.path.exists(args.images):
        with open(args.images, "r", encoding="utf-8") as f:
            selected = {ln.strip() for ln in f if ln.strip()}

    pngs = sorted(glob.glob(os.path.join(args.src_dir, "*.png")))
    pngs = [p for p in pngs if not os.path.basename(p).endswith("_mask.png")]
    if args.limit:
        pngs = pngs[: args.limit]

    n_ok = 0
    for p in pngs:
        stem = os.path.splitext(p)[0]
        base = os.path.basename(stem)
        if selected is not None and base not in selected:
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERR] cannot read {p}")
            continue
        h, w = img.shape[:2]
        up = cv2.resize(img, (int(w * SCALE), int(h * SCALE)),
                        interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(args.out_dir, base + ".png"), up)

        # copy data-domain GT sidecars unchanged
        for ext in (".csv", "_meta.json", "_curves.json"):
            src = stem + ext
            if os.path.exists(src):
                with open(src, "rb") as fi, open(os.path.join(args.out_dir, base + ext), "wb") as fo:
                    fo.write(fi.read())
        # per-curve csvs (multi): stem_cN.csv
        for csv_path in glob.glob(stem + "_c*.csv"):
            cb = os.path.basename(csv_path)
            with open(csv_path, "rb") as fi, open(os.path.join(args.out_dir, cb), "wb") as fo:
                fo.write(fi.read())
        # scaled label boxes
        src_l = stem + "_labels.json"
        if os.path.exists(src_l):
            scale_boxes(src_l, os.path.join(args.out_dir, base + "_labels.json"))
        # scaled exact GT pixels
        src_m = stem + "_mcg.json"
        if os.path.exists(src_m):
            scale_mcg(src_m, os.path.join(args.out_dir, base + "_mcg.json"))
        n_ok += 1

    print(f"[prep] {n_ok} images upsampled x{SCALE} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
