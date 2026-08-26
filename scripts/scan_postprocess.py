"""Post-processing parameter sweep for the multi-curve pipeline.

Loads the segmenter ONCE, then evaluates many cfg overrides on the val
set (fast: ~95s per config on 180 images, model load amortized).

Usage:
  python scripts/scan_postprocess.py --model models/checkpoints/unet_multi_chain.pt \
      --out-dir data/eval_postproc_scan --size 512
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves_multi
from mci.pipeline.extractor import DEFAULTS, load_config
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


def eval_cfg(images, segmenter, cfg, name) -> dict:
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
            recall_6a = recalled / n_gt if n_gt else 1.0
            recall_6b = recalled / max(n_gt, n_pred) if max(n_gt, n_pred) else 1.0
            row.update({"n_gt": n_gt, "n_pred": n_pred, "recalled": recalled,
                        "recall_6a": round(recall_6a, 4), "recall_6b": round(recall_6b, 4),
                        "n_curves_ok": int(n_gt == n_pred),
                        "per_gt_rel": json.dumps([round(r, 4) for r in per_gt])})
        except Exception as e:
            row.update({"n_gt": -1, "n_pred": -1, "recalled": -1,
                        "recall_6a": 0.0, "recall_6b": 0.0,
                        "error": f"{type(e).__name__}: {e}"})
        rows.append(row)
    ok = [r for r in rows if r["n_gt"] > 0]
    total_gt = sum(r["n_gt"] for r in ok)
    total_pred = sum(r["n_pred"] for r in ok)
    total_recalled = sum(r["recalled"] for r in ok)
    r6a = total_recalled / total_gt if total_gt else 0.0
    r6b = total_recalled / max(total_gt, total_pred) if max(total_gt, total_pred) else 0.0
    n_curves_ok = sum(r["n_curves_ok"] for r in ok)
    # bucket counts
    b = {"ok": 0, "e1_2": 0, "e2_5": 0, "e5p": 0}
    for r in ok:
        rels = [float(x) for x in json.loads(r["per_gt_rel"])]
        for rel in rels:
            if rel <= 0.01:
                b["ok"] += 1
            elif rel <= 0.02:
                b["e1_2"] += 1
            elif rel <= 0.05:
                b["e2_5"] += 1
            else:
                b["e5p"] += 1
    res = {"name": name, "n_gt": total_gt, "n_pred": total_pred,
           "recalled": total_recalled, "recall_6a": round(r6a, 4),
           "recall_6b": round(r6b, 4), "n_curves_ok": n_curves_ok,
           "buckets": b, "rows": rows}
    print(f"[{name}] 6a={r6a:.4f} 6b={r6b:.4f} curves_ok={n_curves_ok}/{len(ok)} "
          f"buckets={b}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--model", default="models/checkpoints/unet_multi_chain.pt")
    ap.add_argument("--out-dir", default="data/eval_postproc_scan")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cfg-file", default=None,
                    help="JSON file with list of cfg override dicts (default: built-in sweep)")
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

    base_cfg = load_config()
    segmenter = MultiUNetSegmenter(args.model, size=args.size)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cfg_file:
        with open(args.cfg_file, encoding="utf-8") as f:
            sweep = json.load(f)
    else:
        sweep = [
            {"name": "baseline", "cfg": {}},
            # bias_y sweep (S1b calibration)
            {"name": "bias_-0.4", "cfg": {"multi_refine_bias_y": -0.4}},
            {"name": "bias_-0.5", "cfg": {"multi_refine_bias_y": -0.5}},
            {"name": "bias_-0.8", "cfg": {"multi_refine_bias_y": -0.8}},
            {"name": "bias_-0.9", "cfg": {"multi_refine_bias_y": -0.9}},
            # refine radius sweep (P1 approach-zone skip window)
            {"name": "r4", "cfg": {"multi_refine_radius": 4}},
            {"name": "r8", "cfg": {"multi_refine_radius": 8}},
            {"name": "r10", "cfg": {"multi_refine_radius": 10}},
            # mask threshold
            {"name": "thr0.25", "cfg": {"multi_mask_thr": 0.25}},
            {"name": "thr0.35", "cfg": {"multi_mask_thr": 0.35}},
            # min_area
            {"name": "ma300", "cfg": {"multi_min_area": 300}},
            {"name": "ma600", "cfg": {"multi_min_area": 600}},
        ]

    results = []
    for item in sweep:
        cfg = dict(base_cfg)
        cfg.update(item["cfg"])
        res = eval_cfg(images, segmenter, cfg, item["name"])
        results.append(res)
        with open(os.path.join(args.out_dir, f"{item['name']}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)

    # combined summary
    summary = [{"name": r["name"], "recall_6a": r["recall_6a"], "recall_6b": r["recall_6b"],
                "n_curves_ok": r["n_curves_ok"], "buckets": r["buckets"]} for r in results]
    print("\n================ sweep summary ================")
    for s in sorted(summary, key=lambda s: -s["recall_6a"]):
        print(f"{s['name']:12s} 6a={s['recall_6a']:.4f} 6b={s['recall_6b']:.4f} "
              f"curves_ok={s['n_curves_ok']} buckets={s['buckets']}")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
