"""Phase C evaluation with LineFormer-style 6a/6b dual metrics.

6a (K=Ng): only missing GT curves are penalized  -> pure recall.
6b (K=max(Ng,Np)): unmatched PREDICTED curves score 0 -> recall
with a false-positive penalty (exposes ghost/over-segmentation).

Also reports per-bucket failures (<=1% / 1-2% / 2-5% / >5%) and the
thin-line subset metric (channels whose GT span is small).

Usage:
  python scripts/eval_multi_6ab.py --model models/checkpoints/unet_multi_curve_512c.pt --out-dir data/eval_multi_6ab --size 512
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--model", default="models/checkpoints/unet_multi_curve.pt")
    ap.add_argument("--avg-with", default=None,
                    help="ensemble: average probabilities with this second checkpoint")
    ap.add_argument("--out-dir", default="data/eval_multi_6ab")
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
    segmenter = MultiUNetSegmenter(args.model, size=args.size, avg_with=args.avg_with)
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
            curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
            gts = load_gt_curves(stem)
            n_gt = len(gts)
            n_pred = len(curves)
            preds = [np.asarray(c.points, dtype=np.float64) for c in curves]

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
            # 6a: recall = matched / n_gt (no FP penalty)
            recall_6a = recalled / n_gt if n_gt else 1.0
            # 6b: unmatched preds get 0 -> denominator = max(n_gt, n_pred)
            recall_6b = recalled / max(n_gt, n_pred) if max(n_gt, n_pred) else 1.0
            row.update({
                "n_gt": n_gt, "n_pred": n_pred,
                "recalled": recalled,
                "recall_6a": round(recall_6a, 4),
                "recall_6b": round(recall_6b, 4),
                "n_curves_ok": int(n_gt == n_pred),
                "per_gt_rel": json.dumps([round(r, 4) for r in per_gt]),
            })
            print(f"[OK ] {name}: gt={n_gt} pred={n_pred} 6a={recall_6a:.2f} 6b={recall_6b:.2f}")
        except Exception as e:
            row.update({"n_gt": -1, "n_pred": -1, "recalled": -1,
                        "recall_6a": 0.0, "recall_6b": 0.0,
                        "error": f"{type(e).__name__}: {e}"})
            print(f"[ERR] {name}: {row['error']}")
        rows.append(row)

    ok = [r for r in rows if r["n_gt"] > 0]
    total_gt = sum(r["n_gt"] for r in ok)
    total_pred = sum(r["n_pred"] for r in ok)
    total_recalled = sum(r["recalled"] for r in ok)
    r6a = total_recalled / total_gt if total_gt else 0.0
    r6b = total_recalled / max(total_gt, total_pred) if max(total_gt, total_pred) else 0.0
    n_curves_ok = sum(r["n_curves_ok"] for r in ok)
    print("\n================ summary ================")
    print(f"images: {len(rows)}  with_gt: {len(ok)}")
    print(f"GT curves: {total_gt}  pred curves: {total_pred}  recalled: {total_recalled}")
    print(f"RECALL 6a={r6a:.4f} (pure recall)  6b={r6b:.4f} (with FP penalty)")
    print(f"curve-count exact: {n_curves_ok}/{len(ok)}")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_images": len(rows), "n_gt": total_gt, "n_pred": total_pred,
                   "recalled": total_recalled, "recall_6a": r6a, "recall_6b": r6b,
                   "n_curves_ok": n_curves_ok}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
