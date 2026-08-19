"""Phase C diagnostic: per-image, per-GT rel_rmse + failure classification.

Usage:
  python scripts/eval_multi_diag.py --model models/checkpoints/unet_multi_curve_512c.pt --out-dir data/eval_multi_diag
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
    ap.add_argument("--out-dir", default="data/eval_multi_diag")
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
        name = os.path.basename(img_path)
        row = {"image": name}
        meta = {}
        mp = stem + "_meta.json"
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        row["template"] = meta.get("template_id", "?")
        row["shape"] = meta.get("curve_shape", "?")
        row["deg"] = ",".join(meta.get("degradations", []))
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
                # also record which pred matched (channel) for attribution analysis
                pj = next((j for gi, j, _ in pairs if gi == i), None)
                per_gt.append({"label": label, "rel": round(rel, 4),
                               "pred_chan": pj, "y_span": round(y_span, 5)})
            recalled = sum(1 for r in per_gt if r["rel"] <= 0.01)
            recall = recalled / n_gt if n_gt else 1.0
            row.update({
                "n_gt": n_gt, "n_pred": n_pred,
                "recalled": recalled, "recall": round(recall, 4),
                "n_curves_ok": int(n_gt == n_pred),
                "per_gt": per_gt,
                "buckets": {
                    "ok": sum(1 for r in per_gt if r["rel"] <= 0.01),
                    "e1_2": sum(1 for r in per_gt if 0.01 < r["rel"] <= 0.02),
                    "e2_5": sum(1 for r in per_gt if 0.02 < r["rel"] <= 0.05),
                    "e5p": sum(1 for r in per_gt if r["rel"] > 0.05 or r["rel"] == float("inf")),
                },
            })
            print(f"[OK ] {name}: gt={n_gt} pred={n_pred} recall={recall:.2f} "
                  f"buckets={row['buckets']}")
        except Exception as e:
            row.update({"n_gt": -1, "n_pred": -1, "recalled": -1,
                        "recall": 0.0, "error": f"{type(e).__name__}: {e}"})
            print(f"[ERR] {name}: {row['error']}")
        rows.append(row)

    ok = [r for r in rows if r["n_gt"] > 0]
    total_gt = sum(r["n_gt"] for r in ok)
    total_recalled = sum(r["recalled"] for r in ok)
    b = {"ok": 0, "e1_2": 0, "e2_5": 0, "e5p": 0}
    for r in ok:
        for k in b:
            b[k] += r["buckets"][k]
    print("\n================ summary ================")
    print(f"images: {len(rows)}  with_gt: {len(ok)}  GT curves: {total_gt}")
    print(f"recalled: {total_recalled}  RECALL={total_recalled / total_gt:.4f}")
    print(f"buckets: ok={b['ok']} 1-2%={b['e1_2']} 2-5%={b['e2_5']} >5%={b['e5p']}")
    print(f"curve-count exact: {sum(r['n_curves_ok'] for r in ok)}/{len(ok)}")
    with open(os.path.join(args.out_dir, "diag.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": {"images": len(rows), "with_gt": len(ok),
                               "n_gt": total_gt, "recalled": total_recalled,
                               "recall": total_recalled / total_gt if total_gt else 0,
                               "buckets": b,
                               "n_curves_ok": sum(r["n_curves_ok"] for r in ok)},
                   "rows": rows}, f, indent=1, ensure_ascii=False)
    # bucket-level template / shape breakdown for the >5% class
    from collections import Counter
    for cls in ("e5p", "e2_5"):
        tpl = Counter(); shp = Counter()
        for r in ok:
            for g in r["per_gt"]:
                if (g["rel"] > 0.05 and cls == "e5p") or (0.02 < g["rel"] <= 0.05 and cls == "e2_5"):
                    tpl[r["template"]] += 1
                    shp[r["shape"]] += 1
        print(f"{cls}: templates={dict(tpl)} shapes={dict(shp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
