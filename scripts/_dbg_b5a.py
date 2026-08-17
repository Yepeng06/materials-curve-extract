"""B-5a diagnostic: dump OCR tick readings (text/score/parsed) vs GT values.

Run:  F:\\anaconda3\\envs\\mci\\python.exe scripts\\_dbg_b5a.py --data-dir data/eval_platform
      --images img_0038.png img_0056.png ...   (or --from-report data/eval_b3h_paddle/report.csv --top 30)

Per image prints:
  * structure (plot bbox, axis rows, tick px)
  * x/y axis: GT values | OCR texts (score) -> parsed values
  * whole-image OCR boxes that look numeric but were not used
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from mci.pipeline.chart_structure import detect_structure  # noqa: E402
from mci.pipeline.extractor import load_config  # noqa: E402
from mci.pipeline.tick_reader import (  # noqa: E402
    PaddleOCRBackend,
    _classify_labels,
    _ocr_strips,
    read_ticks,
)
from mci.utils import parse_number_text, read_image  # noqa: E402


def fmt(v):
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/eval_platform")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--from-report", default="")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    images = list(args.images)
    if args.from_report and not images:
        rows = list(csv.DictReader(open(args.from_report, encoding="utf-8")))
        rows.sort(key=lambda r: float(r["rel_rmse"]), reverse=True)
        if args.top:
            rows = rows[: args.top]
        images = [r["image"] for r in rows]

    cfg = load_config()
    ocr = PaddleOCRBackend(lang="en", device="auto")
    for name in images:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            print(f"!! missing {path}")
            continue
        meta_p = os.path.splitext(path)[0] + "_meta.json"
        gt = {}
        if os.path.exists(meta_p):
            with open(meta_p, encoding="utf-8") as f:
                m = json.load(f)
            gt = {
                "x_kind": m.get("x_kind"),
                "y_kind": m.get("y_kind"),
                "x_vals": m.get("x_tick_values", []),
                "y_vals": m.get("y_tick_values", []),
            }
        img = read_image(path)
        structure = detect_structure(img, cfg)
        print("=" * 78)
        print(name, "| plot", structure.plot_bbox,
              "| x_axis_row", structure.x_axis_pixel,
              "| y_axis_col", structure.y_axis_pixel)
        print("  x_ticks_px:", [round(p, 1) for p in structure.x_ticks_px])
        print("  y_ticks_px:", [round(p, 1) for p in structure.y_ticks_px])

        # strip boxes (what the fast path sees)
        strip_boxes = _ocr_strips(img, structure, ocr, cfg)
        xl, yl = _classify_labels(strip_boxes, structure)

        def dump_axis(axis, labels, gt_vals, gt_kind):
            print(f"  [{axis}] GT kind={gt_kind} values={[fmt(v) for v in gt_vals]}")
            if not labels:
                print(f"  [{axis}] strip OCR: NO labels classified")
            for b in labels:
                print(f"  [{axis}] strip  '{b.text}' score={b.score:.3f} "
                      f"center=({b.center[0]:.0f},{b.center[1]:.0f}) -> {fmt(parse_number_text(b.text))}")
            return labels

        dump_axis("x", xl, gt.get("x_vals", []), gt.get("x_kind"))
        dump_axis("y", yl, gt.get("y_vals", []), gt.get("y_kind"))

        # final ticks via read_ticks (includes whole-image fallback)
        try:
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            for axis, tks in (("x", x_ticks), ("y", y_ticks)):
                used = [(round(t.pixel, 1), t.text, fmt(t.value), round(t.score, 3))
                        for t in tks]
                print(f"  [{axis}] final ticks (px,text,val,score): {used}")
        except Exception as e:
            print(f"  read_ticks FAILED: {type(e).__name__}: {e}")

        # whole-image OCR numeric boxes (candidates the strips may have missed)
        full = ocr.read_text_boxes(img)
        used_centers = set()
        for b in list(xl) + list(yl):
            used_centers.add((round(b.center[0] / 10), round(b.center[1] / 10)))
        numeric_extra = []
        for b in full:
            v = parse_number_text(b.text)
            if v is None:
                continue
            if (round(b.center[0] / 10), round(b.center[1] / 10)) in used_centers:
                continue
            numeric_extra.append((b.text, round(b.score, 3),
                                  (round(b.center[0]), round(b.center[1])), fmt(v)))
        if numeric_extra:
            print(f"  [full] numeric boxes NOT in strips: {numeric_extra[:14]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
