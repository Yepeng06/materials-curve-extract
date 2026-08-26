"""S3: A/B/C quality-gate triage on the R1e 100-image real subset.

Runs the full pipeline (structure -> paddle ticks -> axes -> multi-curve
extraction) and classifies EVERY image with the quality gate
(REAL_ROBUSTNESS_DESIGN.md): success -> A, explainable failure -> B,
unsupported -> C.  Reports the triage distribution + reject-code breakdown,
stratified by source (chartub / pmc / chartqa).

Usage:
  python scripts/eval_real_robust.py [--subset ...] [--out ...] [--limit N]
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
from mci.pipeline.panel_detect import detect_panels
from mci.pipeline.quality_gate import (
    classify_failure,
    is_categorical_axis,
)
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.tick_reader import PaddleOCRBackend, read_ticks
from mci.schema import AxisFitError, ExtractionError
from mci.utils import read_image


def load_manifest() -> dict:
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
    key = key.rsplit(".", 1)[0]
    if key in manifest:
        rel = manifest[key].get("orig_rel", "")
        cand = os.path.join("data", "real_diag", rel)
        if os.path.exists(cand):
            return cand
    base = os.path.join("data", "real_diag")
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.rsplit(".", 1)[0] == key and fn.lower().endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(root, fn)
    return None


def alpha_composite_white(img_bgr: np.ndarray) -> np.ndarray:
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
    ap.add_argument("--out", default=os.path.join("data", "experiments_r1e", "robust_triage.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        keys = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        keys = keys[: args.limit]

    rows = []
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out, encoding="utf-8"))
            rows = list(prev.get("rows", []))
        except Exception:
            rows = []
    done = {r["image"] for r in rows}

    cfg = load_config()
    ocr = PaddleOCRBackend(lang="en", device="auto")
    segmenter = MultiUNetSegmenter(cfg.get("multi_unet_checkpoint"),
                                   size=int(cfg.get("unet_size", 512)))
    manifest = load_manifest()

    for key in keys:
        if key in done:
            continue
        img_path = resolve_image_path(key, manifest)
        if img_path is None:
            rows.append({"image": key, "quality": "C", "status": "rejected",
                         "reject_code": "IMG_NOT_FOUND", "error": "not found"})
            continue
        t0 = time.time()
        row = {"image": key, "source": key.split("_")[0]}
        try:
            img = alpha_composite_white(read_image(img_path))
            panels = detect_panels(img)
            row["panels"] = len(panels)
            structure = detect_structure(img, cfg)
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            nv_x = sum(1 for t in x_ticks if t.value is not None)
            nv_y = sum(1 for t in y_ticks if t.value is not None)
            row["n_ticks"] = [nv_x, nv_y]
            try:
                x_axis, y_axis = build_axes(
                    x_ticks, y_ticks,
                    x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
                    y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
                )
            except AxisFitError:
                if is_categorical_axis(x_ticks) or is_categorical_axis(y_ticks):
                    row.update({"quality": "B", "status": "partial",
                                "reject_code": "OCR_CATEGORICAL_AXIS",
                                "error": "categorical axis"})
                else:
                    v = classify_failure(img, error=AxisFitError("few ticks"),
                                         n_valid_ticks=nv_x + nv_y)
                    row.update({"quality": v.quality, "status": v.status,
                                "reject_code": v.reject_code,
                                "error": "AxisFitError"})
                rows.append(row)
                continue
            curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
            worst_q = min(x_axis.quality, y_axis.quality)
            if worst_q < 0.95:
                row.update({"quality": "B", "status": "ok",
                            "reject_code": "AXIS_TYPE_AMBIGUOUS",
                            "n_curves": len(curves), "axis_q": round(worst_q, 4)})
            else:
                row.update({"quality": "A", "status": "ok", "n_curves": len(curves),
                            "axis_q": round(worst_q, 4)})
        except ExtractionError as e:
            v = classify_failure(img, error=e)
            row.update({"quality": v.quality, "status": v.status,
                        "reject_code": v.reject_code,
                        "error": f"{type(e).__name__}: {e}"[:160]})
        except Exception as e:  # unexpected -> conservative C
            row.update({"quality": "C", "status": "rejected",
                        "reject_code": "CURVE_EXTRACT_FAIL",
                        "error": f"{type(e).__name__}: {e}"[:160]})
        row["secs"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{row['quality']}] {key}: {row.get('reject_code', 'ok')} "
              f"panels={row.get('panels', '?')} ({row['secs']}s)", flush=True)
        if len(rows) % 10 == 0:
            _save(args.out, rows)

    _save(args.out, rows)
    _report(rows)
    return 0


def _save(out: str, rows: list) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _report(rows: list) -> None:
    from collections import Counter

    n = len(rows)
    q = Counter(r["quality"] for r in rows)
    codes = Counter(r.get("reject_code") or "OK" for r in rows)
    print("\n================ robust triage summary ================")
    print(f"images: {n}")
    for tier in ("A", "B", "C"):
        print(f"  {tier}: {q.get(tier, 0)} ({q.get(tier, 0) / max(1, n):.1%})")
    print("reject codes:", dict(codes.most_common()))
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], Counter())[r["quality"]] += 1
    for src, c in sorted(by_src.items()):
        print(f"  {src}: A={c.get('A', 0)} B={c.get('B', 0)} C={c.get('C', 0)}")


if __name__ == "__main__":
    sys.exit(main())
