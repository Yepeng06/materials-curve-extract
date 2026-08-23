"""R1e: zero-shot out-of-domain diagnosis on real chart images.

Runs the full production pipeline (YOLO structure -> PaddleOCR ticks ->
axis fit -> GOI multi-curve extraction) on the 100-image real subset
(data/real_diag/images/eval_subset.txt) WITHOUT any fine-tuning, and
records per-image success / axis-kind / curve count / failure reason.

Real images have no *_labels.json sidecars, so stub OCR is unusable:
this script uses the PaddleOCR backend (slow, ~6-15s/image CPU) and the
default config (GOI + independent + min_area450 + embed_merge).

Usage:
  python scripts/eval_real_zero_shot.py [--subset data/real_diag/images/eval_subset.txt]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves_multi
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.tick_reader import PaddleOCRBackend, read_ticks
from mci.utils import read_image


def load_manifest() -> dict:
    """image key (no ext) -> {source, orig_rel, width, height}."""
    out = {}
    path = os.path.join("data", "real_diag", "images", "manifest.csv")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = r["image"].rsplit(".", 1)[0]
            out[key] = r
    return out


def resolve_image_path(key: str, manifest: dict) -> str | None:
    """manifest.csv orig_rel is relative to data/real_diag/."""
    key = key.rsplit(".", 1)[0]  # tolerate '.jpg'/'.png' suffixes in the subset list
    if key in manifest:
        rel = manifest[key].get("orig_rel", "")
        cand = os.path.join("data", "real_diag", rel)
        if os.path.exists(cand):
            return cand
    # fallbacks by prefix
    base = os.path.join("data", "real_diag")
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.rsplit(".", 1)[0] == key and fn.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(root, fn)
    return None


def alpha_composite_white(img_bgr: np.ndarray) -> np.ndarray:
    """ChartQA PNGs are RGBA; composite onto white before any processing."""
    if img_bgr.ndim == 2:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if img_bgr.shape[2] == 4:
        b, g, r, a = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2], img_bgr[:, :, 3] / 255.0
        a = a[..., None]
        white = np.full((h, w, 3), 255, dtype=np.float32)
        comp = white * (1 - a) + np.stack([b, g, r], axis=2) * a
        return comp.astype(np.uint8)
    return img_bgr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default=os.path.join("data", "real_diag", "images", "eval_subset.txt"))
    ap.add_argument("--out", default=os.path.join("data", "experiments_r1e", "zero_shot_diag.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        keys = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        keys = keys[: args.limit]

    # resume: skip images already processed when the previous run crashed
    rows = []
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out, encoding="utf-8"))
            rows = list(prev.get("rows", []))
        except Exception:
            rows = []
    done = {r["key"] for r in rows}

    manifest = load_manifest()
    cfg = load_config()
    segmenter = MultiUNetSegmenter(cfg.get("multi_unet_checkpoint", "models/checkpoints/unet_multi_goi.pt"),
                                   size=int(cfg.get("unet_size", 512)))
    ocr = PaddleOCRBackend()

    for key in keys:
        if key in done:
            continue
        ipath = resolve_image_path(key, manifest)
        row = {"key": key, "path": ipath}
        if ipath is None:
            row.update(status="no_image", elapsed_s=0.0)
            rows.append(row)
            print(f"[NOIMG] {key}")
            continue
        t0 = time.time()
        try:
            img = read_image(ipath)
            img = alpha_composite_white(img)
            structure = detect_structure(img, cfg)
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            # T2b: categorical axis detection -- an axis whose labels are all
            # non-numeric (drug names, month names...) carries no numeric
            # scale; bar charts / category plots are NOT curve-extraction
            # targets and must not be counted as pipeline failures.
            from mci.utils import parse_number_text

            def _axis_kind(ticks, side):
                vals = [parse_number_text(t.text) for t in ticks if t.text]
                if len(vals) >= 2 and all(v is None for v in vals):
                    return "categorical"
                return "numeric"

            xk = _axis_kind(x_ticks, "x")
            yk = _axis_kind(y_ticks, "y")
            if xk == "categorical" or yk == "categorical":
                row.update(status="categorical", x_axis=xk, y_axis=yk,
                           n_x_ticks=len(x_ticks), n_y_ticks=len(y_ticks),
                           elapsed_s=round(time.time() - t0, 2))
                rows.append(row)
                print(f"[CAT] {key}: x={xk} y={yk}")
                continue
            x_axis, y_axis = build_axes(
                x_ticks, y_ticks,
                x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
                y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
            )
            curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
            row.update(status="ok",
                       n_curves=len(curves),
                       x_kind=x_axis.kind.value,
                       y_kind=y_axis.kind.value,
                       n_x_ticks=len(x_ticks),
                       n_y_ticks=len(y_ticks),
                       plot_w=int(structure.plot_bbox[2] - structure.plot_bbox[0] + 1),
                       plot_h=int(structure.plot_bbox[3] - structure.plot_bbox[1] + 1),
                       n_points=sum(len(c.points) for c in curves),
                       elapsed_s=round(time.time() - t0, 2))
            print(f"[OK ] {key}: curves={len(curves)} xk={row['x_kind']} yk={row['y_kind']} "
                  f"ticks=({len(x_ticks)},{len(y_ticks)}) {row['elapsed_s']}s")
        except Exception as e:
            row.update(status="fail", error=f"{type(e).__name__}: {str(e)[:150]}",
                       elapsed_s=round(time.time() - t0, 2))
            print(f"[ERR] {key}: {row['error']}")
        rows.append(row)

    ok = [r for r in rows if r["status"] == "ok"]
    fail = [r for r in rows if r["status"] == "fail"]
    from collections import Counter
    axis_kinds = Counter()
    for r in ok:
        axis_kinds[f"{r.get('x_kind')}-{r.get('y_kind')}"] += 1
    summary = {
        "n_total": len(rows), "n_ok": len(ok), "n_fail": len(fail),
        "fail_reasons": Counter(r.get("error", "").split(":")[0] for r in fail),
        "curve_count_hist": Counter(r["n_curves"] for r in ok),
        "axis_kinds": axis_kinds,
        "median_elapsed_s": round(float(np.median([r["elapsed_s"] for r in rows])), 2) if rows else None,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1, ensure_ascii=False)
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
