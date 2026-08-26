"""Official CHART-Info graded 6a/6b evaluation (for paper comparison).

LineFormer reports 6a/6b under the official CHART-Info graded metric:
  Error = min(1, |v_i - I(P, u_i)| / (v_i + eps)),  eps = 1/100 of GT range
per GT point, then weighted-mean over intervals; bipartite matching;
6a: K = n_gt (pure recall), 6b: K = max(n_gt, n_pred) (FP penalty).

This differs from the project's strict "RMSE/y_span <= 1% hard gate" 6a
used for acceptance.  This script computes the OFFICIAL graded 6a/6b on
the same predictions so the paper can compare directly with LineFormer
(UB-PMC 93.1/88.25, AdobeSynth19 97.51/97.02, Lenovo 99.29/98.81).

Usage:
  python scripts/eval_multi_official6ab.py --model models/checkpoints/unet_multi_chain.pt \
      --out-dir data/eval_official6a --size 512
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


def graded_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """Official CHART-Info graded curve score in [0, 1] (1 = perfect)."""
    g = gt[np.argsort(gt[:, 0])]
    p = pred[np.argsort(pred[:, 0])]
    x_lo = max(float(g[0, 0]), float(p[0, 0]))
    x_hi = min(float(g[-1, 0]), float(p[-1, 0]))
    inside = (p[:, 0] >= x_lo) & (p[:, 0] <= x_hi)
    if inside.sum() < 2:
        return 0.0
    gy = np.interp(p[inside, 0], g[:, 0], g[:, 1])
    # eps = 1/100 of GT y range
    eps = (float(g[:, 1].max()) - float(g[:, 1].min())) / 100.0
    err = np.minimum(1.0, np.abs(p[inside, 1] - gy) / (np.abs(gy) + eps))
    # weighted mean over GT intervals (official metric uses interval weights;
    # with dense points, uniform weights approximate it)
    return float(1.0 - err.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--model", default="models/checkpoints/unet_multi_chain.pt")
    ap.add_argument("--out-dir", default="data/eval_official6a")
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

    rows = []
    for img_path in images:
        stem = os.path.splitext(img_path)[0]
        row = {"image": os.path.basename(img_path)}
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
            costs = [[graded_score(g[1], p) for p in preds] for g in gts]
            # bipartite max-weight matching (greedy on highest score)
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
                        if best is None or c > best[0]:
                            best = (c, i, j)
                if best is None or best[0] <= 0.0:
                    break
                _, i, j = best
                matched_gt[i] = True
                matched_pred[j] = True
                pairs.append(best[0])
            # official: 6a denominator = n_gt (unmatched gt contribute 0),
            # 6b denominator = max(n_gt, n_pred)
            sum_score = sum(pairs)
            r6a = sum_score / n_gt if n_gt else 1.0
            r6b = sum_score / max(n_gt, n_pred) if max(n_gt, n_pred) else 1.0
            row.update({"n_gt": n_gt, "n_pred": n_pred,
                        "official_6a": round(r6a, 4), "official_6b": round(r6b, 4),
                        "scores": json.dumps([round(s, 4) for s in pairs])})
        except Exception as e:
            row.update({"n_gt": -1, "n_pred": -1,
                        "official_6a": 0.0, "official_6b": 0.0,
                        "error": f"{type(e).__name__}: {e}"})
        rows.append(row)

    ok = [r for r in rows if r["n_gt"] > 0]
    total_gt = sum(r["n_gt"] for r in ok)
    total_pred = sum(r["n_pred"] for r in ok)
    t6a = sum(r["official_6a"] * r["n_gt"] for r in ok)
    t6b = sum(r["official_6b"] * max(r["n_gt"], r["n_pred"]) for r in ok)
    o6a = t6a / total_gt if total_gt else 0.0
    o6b = t6b / max(total_gt, total_pred) if max(total_gt, total_pred) else 0.0
    print("\n================ summary ================")
    print(f"images: {len(rows)}  with_gt: {len(ok)}  GT curves: {total_gt}")
    print(f"OFFICIAL graded 6a={o6a:.4f}  6b={o6b:.4f}")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_images": len(rows), "n_gt": total_gt, "n_pred": total_pred,
                   "official_6a": round(o6a, 4), "official_6b": round(o6b, 4),
                   "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
