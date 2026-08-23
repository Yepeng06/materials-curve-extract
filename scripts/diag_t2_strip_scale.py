"""T2 minimal experiment: does upscaling the axis label strip improve
PaddleOCR tick reading on real paper figures (the 20 AxisFitError cases)?

For each failing real image (from R1e zero-shot diag): crop the bottom
(x) and left (y) label strips, OCR them at 1x / 2x / 4x (Lanczos), count
readable numeric labels and compare.

Usage: python scripts/diag_t2_strip_scale.py
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np

from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.extractor import load_config
from mci.pipeline.tick_reader import PaddleOCRBackend
from mci.utils import read_image

NUM_RE = re.compile(r"^[-+.]?[0-9][0-9,.]*$")


def count_numeric(boxes) -> int:
    return sum(1 for b in boxes if NUM_RE.match(b.text.strip()))


def main() -> int:
    cfg = load_config()
    ocr = PaddleOCRBackend()
    diag = json.load(open("data/experiments_r1e/zero_shot_diag.json", encoding="utf-8"))
    fails = [r for r in diag["rows"] if r["status"] == "fail" and "AxisFit" in r.get("error", "")]

    rows = []
    for r in fails[:20]:
        ipath = r["path"]
        if not ipath or not os.path.exists(ipath):
            continue
        name = os.path.basename(ipath)
        try:
            img = read_image(ipath)
            if img.ndim == 3 and img.shape[2] == 4:
                a = img[:, :, 3] / 255.0
                img = (img[:, :, :3] * a[..., None] + 255 * (1 - a[..., None])).astype(np.uint8)
            structure = detect_structure(img, cfg)
        except Exception as e:
            rows.append({"image": name, "error": f"structure: {e}"})
            continue
        x0, y0, x1, y1 = structure.plot_bbox
        h, w = img.shape[:2]
        # label strips: bottom band below plot, left band left of plot
        strips = {}
        if y1 + 5 < h:
            strips["x"] = img[max(0, y1): min(h, y1 + 90), max(0, x0 - 40): min(w, x1 + 40)]
        if x0 - 5 > 0:
            strips["y"] = img[max(0, y0 - 20): min(h, y1 + 20), max(0, x0 - 100): x0]

        row = {"image": name}
        for axis, strip in strips.items():
            if strip.size == 0:
                continue
            for scale in (1, 2, 4):
                up = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
                boxes = ocr.read_text_boxes(up)
                n = count_numeric(boxes)
                texts = [b.text for b in boxes]
                row[f"{axis}_s{scale}_n"] = n
                if scale == 1:
                    row[f"{axis}_s1_texts"] = texts[:12]
            row[f"{axis}_gt_ticks"] = "?"
        rows.append(row)
        print(f"[OK ] {name}: " + " ".join(
            f"{a}={row.get(a + '_s1_n', '-')}/{row.get(a + '_s2_n', '-')}/{row.get(a + '_s4_n', '-')}"
            for a in ("x", "y") if a in strips))

    os.makedirs("data/experiments_t2", exist_ok=True)
    with open("data/experiments_t2/strip_scale_diag.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
