"""S3 validation: how robust is the current axis calibration (LS + RANSAC +
log-domain fit, coordinate_mapper.py) to tick perturbations?

Three variants on val_multi (stub OCR sidecars):
  A. baseline build_axes
  B. all tick pixels shifted +1 px (systematic 1-px scale offset)
  C. one random tick value corrupted x10 (OCR misread, RANSAC should reject)

Metric: for each GT curve, map GT data points -> pixels (value_to_pixel) then
back (pixel_to_value) and measure rel error vs y-span.  The round-trip error
of A is the calibration floor; B/C show error propagation.
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
from mci.pipeline.extractor import load_config
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


def prop_rel(x_ref, y_ref, x_inj, y_inj, gt: np.ndarray) -> float:
    """Propagated calibration error: GT data -> pixels via INJECTED axes ->
    back to data via REFERENCE axes.  Measures how tick errors distort the
    mapping (A vs A is 0 by construction)."""
    y_span = float(gt[:, 1].max() - gt[:, 1].min())
    if y_span <= 0:
        return float("inf")
    px = np.array([x_inj.value_to_pixel(v) for v in gt[:, 0]])
    py = np.array([y_inj.value_to_pixel(v) for v in gt[:, 1]])
    fin = np.isfinite(px) & np.isfinite(py)
    if int(fin.sum()) < 2:
        return float("inf")
    vx = np.array([x_ref.pixel_to_value(p) for p in px[fin]])
    vy = np.array([y_ref.pixel_to_value(p) for p in py[fin]])
    err = np.sqrt(np.mean((vy - gt[fin, 1]) ** 2))
    return float(err / y_span)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--out", default="data/experiments_s3/calib_summary.json")
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
    rows = []
    for img_path in images:
        stem = os.path.splitext(img_path)[0]
        name = os.path.basename(img_path)
        try:
            img = read_image(img_path)
            structure = detect_structure(img, cfg)
            ocr = StubOCRBackend(stem + "_labels.json")
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            gts = load_gt_curves(stem)
            xep = (float(structure.y_axis_pixel), float(structure.plot_bbox[2]))
            yep = (float(structure.plot_bbox[1]), float(structure.x_axis_pixel))

            def build(xt, yt):
                return build_axes(
                    xt, yt,
                    x_endpoints=xep, y_endpoints=yep,
                )

            xa, ya = build(x_ticks, y_ticks)

            # B: +1 px on every tick pixel
            def shift(ticks):
                out = []
                for t in ticks:
                    import dataclasses
                    out.append(dataclasses.replace(t, pixel=t.pixel + 1.0))
                return out

            xb, yb = build(shift(x_ticks), shift(y_ticks))

            # C: corrupt one random tick value x10 (keep pixels)
            def corrupt(ticks):
                import dataclasses
                out = []
                rng = np.random.default_rng(20260823)
                i = rng.integers(0, len(ticks)) if len(ticks) > 2 else -1
                for j, t in enumerate(ticks):
                    v = t.value
                    if j == i and v is not None and abs(v) > 1e-9:
                        out.append(dataclasses.replace(t, value=v * 10.0))
                    else:
                        out.append(t)
                return out

            xc, yc = build(corrupt(x_ticks), corrupt(y_ticks))

            row = {"image": name,
                   "n_xticks": len(x_ticks), "ny_ticks": len(y_ticks),
                   "x_kind": xa.kind.value, "y_kind": ya.kind.value}
            for tag, (xin_, yin_) in (("A", (xa, ya)), ("B", (xb, yb)), ("C", (xc, yc))):
                rels = [prop_rel(xa, ya, xin_, yin_, g[1]) for g in gts]
                row[f"{tag}_rel_median"] = round(float(np.median(rels)), 5) if rels else None
                row[f"{tag}_rel_max"] = round(float(np.max(rels)), 5) if rels else None
            rows.append(row)
        except Exception as e:
            rows.append({"image": name, "error": f"{type(e).__name__}: {e}"})
            print(f"[ERR] {name}: {rows[-1]['error']}")

    ok = [r for r in rows if "A_rel_median" in r]
    stats = {}
    for tag in ("A", "B", "C"):
        meds = np.array([r[f"{tag}_rel_median"] for r in ok if r.get(f"{tag}_rel_median") is not None])
        stats[tag] = {"n": int(len(meds)),
                      "median": round(float(np.median(meds)), 5) if len(meds) else None,
                      "p90": round(float(np.percentile(meds, 90)), 5) if len(meds) else None,
                      "max": round(float(np.max(meds)), 5) if len(meds) else None}
        print(f"{tag}: n={stats[tag]['n']} median={stats[tag]['median']} "
              f"p90={stats[tag]['p90']} max={stats[tag]['max']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "rows": ok}, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
