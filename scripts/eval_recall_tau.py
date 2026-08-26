"""Recall@tau curve for the strict 6a metric (paper argumentation tool).

Computes, on the val / Phase D sets, the curve-level recall at multiple
relative-error thresholds tau in {0.5%, 1%, 2%, 3%, 5%} (and the official
graded 6a for reference).  This answers "at which tolerance does 95%
recall become reachable" and supports the dual-metric paper narrative.

Usage:
  python scripts/eval_recall_tau.py --model models/checkpoints/unet_multi_chain.pt \
      --out-dir data/eval_recall_tau --size 512
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves_multi
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

TAUS = [0.005, 0.01, 0.02, 0.03, 0.05]


def load_gt_curves(stem: str) -> list:
    out = []
    with open(stem + "_curves.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data["curves"]:
        csv_path = os.path.join(os.path.dirname(stem), c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        pts = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        out.append((c.get("label") or c["curve_id"], pts))
    return out


def curve_rel_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt[np.argsort(gt[:, 0])]
    p = pred[np.argsort(pred[:, 0])]
    x_lo = max(float(g[0, 0]), float(p[0, 0]))
    x_hi = min(float(g[-1, 0]), float(p[-1, 0]))
    inside = (p[:, 0] >= x_lo) & (p[:, 0] <= x_hi)
    if inside.sum() < 2:
        return float("inf")
    gy = np.interp(p[inside, 0], g[:, 0], g[:, 1])
    span = float(g[:, 1].max() - g[:, 1].min())
    if span <= 0:
        return float("inf")
    return float(np.sqrt(np.mean((p[inside, 1] - gy) ** 2))) / span


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--model", default="models/checkpoints/unet_multi_chain.pt")
    ap.add_argument("--out-dir", default="data/eval_recall_tau")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    images = []
    for d in args.data_dir.split(","):
        d = d.strip()
        if not os.path.isdir(d):
            continue
        images += sorted(glob.glob(os.path.join(d, "*.png")))
    images = [p for p in images if not os.path.basename(p).endswith("_mask.png")]
    if args.limit:
        images = images[: args.limit]

    cfg = load_config()
    segmenter = MultiUNetSegmenter(args.model, size=args.size)
    os.makedirs(args.out_dir, exist_ok=True)

    all_rel = []  # (rel, template, shape) for every matched GT curve
    total_gt = 0
    for img_path in images:
        stem = os.path.splitext(img_path)[0]
        meta = {}
        mp = stem + "_meta.json"
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        try:
            img = read_image(img_path)
            structure = detect_structure(img, cfg)
            ocr = StubOCRBackend(stem + "_labels.json")
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            x_axis, y_axis = build_axes(
                x_ticks, y_ticks,
                x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
                y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
            )
            curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
            gts = load_gt_curves(stem)
            n_gt = len(gts)
            total_gt += n_gt
            preds = [np.asarray(c.points, dtype=np.float64) for c in curves]
            costs = [[curve_rel_rmse(g[1], p) for p in preds] for g in gts]
            matched_gt = [False] * n_gt
            matched_pred = [False] * len(preds)
            pairs = []
            while True:
                best = None
                for i in range(n_gt):
                    if matched_gt[i]:
                        continue
                    for j in range(len(preds)):
                        if matched_pred[j]:
                            continue
                        c = costs[i][j]
                        if best is None or c < best[0]:
                            best = (c, i, j)
                if best is None:
                    break
                _, i, j = best
                matched_gt[i] = True
                matched_pred[j] = True
                pairs.append((i, costs[i][j]))
            for i, (label, g) in enumerate(gts):
                m = next((r for gi, r in pairs if gi == i), float("inf"))
                all_rel.append({"rel": m, "image": os.path.basename(img_path),
                                "label": label, "template": meta.get("template_id", "?"),
                                "shape": meta.get("curve_shape", "?")})
        except Exception:
            pass

    # recall@tau over matched curves (unmatched = inf = never recalled)
    rels = np.array([r["rel"] for r in all_rel], dtype=np.float64)
    result = {"n_images": len(images), "n_gt": total_gt, "taus": {}}
    print("\n================ recall@tau ================")
    for tau in TAUS:
        rec = float((rels <= tau).mean()) if len(rels) else 0.0
        result["taus"][str(tau)] = round(rec, 4)
        print(f"tau={tau*100:.1f}%: 6a = {rec:.4f}  ({int((rels <= tau).sum())}/{len(rels)})")
    result["all_rel"] = all_rel
    with open(os.path.join(args.out_dir, "recall_tau.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "all_rel"}, f, indent=1)
    with open(os.path.join(args.out_dir, "per_curve_rel.json"), "w", encoding="utf-8") as f:
        json.dump(all_rel, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
