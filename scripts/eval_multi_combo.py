"""Phase C evaluation: multi-curve extraction, combined mode.

Usage:
  python scripts/eval_multi_combo.py --data-dir data/val_multi,data/val_single \
      --model models/checkpoints/unet_multi_curve_v3.pt --combo

Combo mode: the 512-trained SINGLE-curve U-Net provides the precise
semantic mask (all curves, sub-pixel quality), and the multi-curve U-Net
provides per-pixel instance ASSIGNMENT (argmax over channels).  The two
are combined: semantic pixels split by assignment -> per-instance masks.
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from mci.pipeline.chart_structure import detect_structure  # noqa: E402
from mci.pipeline.coordinate_mapper import build_axes  # noqa: E402
from mci.pipeline.curve_extractor import (  # noqa: E402
    _column_centroid,
    _filter_mask_fragments,
    _refine_chain,
    _trace_chain,
    extract_curves_multi,
)
from mci.pipeline.extractor import load_config  # noqa: E402
from mci.pipeline.segmenter import MultiUNetSegmenter, UNetSegmenter  # noqa: E402
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks  # noqa: E402
from mci.utils import read_image  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402


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


def curve_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt[np.argsort(gt[:, 0])]
    p = pred[np.argsort(pred[:, 0])]
    x_lo = max(float(g[0, 0]), float(p[0, 0]))
    x_hi = min(float(g[-1, 0]), float(p[-1, 0]))
    inside = (p[:, 0] >= x_lo) & (p[:, 0] <= x_hi)
    if inside.sum() < 2:
        return float("inf")
    gy = np.interp(p[inside, 0], g[:, 0], g[:, 1])
    return float(np.sqrt(np.mean((p[inside, 1] - gy) ** 2)))


def _combined_curves(image_bgr, structure, x_axis, y_axis, cfg, sem_seg, inst_seg):
    """Semantic mask (precise) split by instance-channel argmax assignment."""
    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    sem_prob = sem_seg.prob_full(image_bgr)
    inst_prob = inst_seg.prob_full(image_bgr)
    reg_sem = sem_prob[y0 : y1 + 1, x0 : x1 + 1]
    reg_inst = inst_prob[:, y0 : y1 + 1, x0 : x1 + 1]
    sem_mask = (reg_sem > 0.5).astype(np.uint8)
    amax = np.argmax(reg_inst, axis=0)
    curves = []
    for c in range(reg_inst.shape[0]):
        m = ((sem_mask > 0) & (amax == c)).astype(np.uint8)
        if int(m.sum()) < 16:
            continue
        m = _filter_mask_fragments(m, plot_w, plot_h)
        if int(m.sum()) < 16:
            continue
        skel = skeletonize(m.astype(bool)).astype(np.uint8)
        chain = _trace_chain(skel)
        if chain is not None:
            xs = [p[0] for p in chain]
            if (max(xs) - min(xs) + 1) < 0.7 * plot_w:
                chain = None
        if chain is not None:
            chain = _refine_chain(reg_sem, chain)
        else:
            chain = _column_centroid(reg_sem)
        if not chain or len(chain) < 8:
            continue
        points = [(x_axis.pixel_to_value(x0 + px), y_axis.pixel_to_value(y0 + py))
                  for px, py in chain]
        curves.append(points)
    return curves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--model", default="models/checkpoints/unet_multi_curve.pt")
    ap.add_argument("--sem-model", default="models/checkpoints/unet_curve.pt")
    ap.add_argument("--out-dir", default="data/eval_multi")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--combo", action="store_true")
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
    if not images:
        print("no images found")
        return 1

    cfg = load_config()
    inst_seg = MultiUNetSegmenter(args.model, size=args.size)
    sem_seg = UNetSegmenter(args.sem_model, size=512) if args.combo else None
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for img_path in images:
        stem = os.path.splitext(img_path)[0]
        name = os.path.basename(img_path)
        row = {"image": name}
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
            if args.combo:
                curves = _combined_curves(img, structure, x_axis, y_axis, cfg,
                                          sem_seg, inst_seg)
            else:
                curves = extract_curves_multi(img, structure, x_axis, y_axis,
                                              cfg, inst_seg)
            gts = load_gt_curves(stem)
            n_gt = len(gts)
            n_pred = len(curves)
            preds = [np.asarray(c, dtype=np.float64) for c in curves]

            costs = [[curve_rmse(g[1], p) for p in preds] for g in gts]
            matched_gt = [False] * n_gt
            matched_pred = [False] * n_pred
            pairs = []
            while True:
                best = None
                for i in range(n_gt):
                    if matched_gt[i]:
                        continue
                    for j in range(n_pred):
                        if matched_pred[j]:
                            continue
                        c = costs[i][j]
                        if c == float("inf"):
                            continue
                        if best is None or c < best[0]:
                            best = (c, i, j)
                if best is None:
                    break
                _, i, j = best
                matched_gt[i] = True
                matched_pred[j] = True
                pairs.append((i, j, best[0]))

            per_gt = []
            for i, (label, g) in enumerate(gts):
                y_span = float(g[:, 1].max() - g[:, 1].min())
                m = next((rmse for gi, _, rmse in pairs if gi == i), float("inf"))
                rel = m / y_span if y_span > 0 else float("inf")
                per_gt.append(rel)
            recalled = sum(1 for r in per_gt if r <= 0.01)
            recall = recalled / n_gt if n_gt else 1.0
            row.update({
                "n_gt": n_gt, "n_pred": n_pred,
                "recalled": recalled, "recall": round(recall, 4),
                "n_curves_ok": int(n_gt == n_pred),
                "per_gt_rel": json.dumps([round(r, 4) for r in per_gt]),
            })
            print(f"[OK ] {name}: gt={n_gt} pred={n_pred} recall={recall:.2f}")
        except Exception as e:
            row.update({"n_gt": -1, "n_pred": -1, "recalled": -1,
                        "recall": 0.0, "error": f"{type(e).__name__}: {e}"})
            print(f"[ERR] {name}: {row['error']}")
        rows.append(row)

    ok = [r for r in rows if r["n_gt"] > 0]
    total_gt = sum(r["n_gt"] for r in ok)
    total_recalled = sum(r["recalled"] for r in ok)
    recall_all = total_recalled / total_gt if total_gt else 0.0
    n_curves_ok = sum(r["n_curves_ok"] for r in ok)
    print("\n================ summary ================")
    print(f"images: {len(rows)}  with_gt: {len(ok)}")
    print(f"GT curves: {total_gt}  recalled: {total_recalled}  "
          f"RECALL={recall_all:.4f}  (target >= 0.95)")
    print(f"curve-count exact: {n_curves_ok}/{len(ok)}")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_images": len(rows), "n_gt": total_gt,
                   "recalled": total_recalled, "recall": recall_all,
                   "n_curves_ok": n_curves_ok}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
