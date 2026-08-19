"""Characterize ghost (over-predicted) channels on 5-pred images.

For each val_multi image where the model emits 5 channels (GT=4), dump
per-channel stats: max prob, pixel area>thr, x-span, y-span, chain length,
and whether the channel matches a GT curve.
"""
from __future__ import annotations
import argparse, csv as csvlib, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves_multi, _filter_mask_fragments
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/checkpoints/unet_multi_curve_512c.pt")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config()
    seg = MultiUNetSegmenter(args.model, size=args.size)
    for p in sorted(glob.glob("data/val_multi/*.png")):
        if p.endswith("_mask.png"):
            continue
        stem = os.path.splitext(p)[0]
        img = read_image(p)
        structure = detect_structure(img, cfg)
        ocr = StubOCRBackend(stem + "_labels.json")
        x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
            y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
        )
        prob = seg.prob_full(img)
        x0, y0, x1, y1 = structure.plot_bbox
        reg = prob[:, y0:y1+1, x0:x1+1]
        n_gt = len(json.load(open(stem + "_curves.json", encoding="utf-8"))["curves"])
        n_active = 0
        stats = []
        for c in range(prob.shape[0]):
            m = reg[c]
            if float(m.max()) < 0.3:
                continue
            n_active += 1
            mask01 = ((m > 0.3) & (np.argmax(reg, axis=0) == c)).astype(np.uint8)
            mask01 = _filter_mask_fragments(mask01, x1-x0+1, y1-y0+1)
            ys, xs = np.nonzero(mask01)
            area = int(mask01.sum())
            xspan = (xs.max()-xs.min()+1) if len(xs) else 0
            yspan = (ys.max()-ys.min()+1) if len(ys) else 0
            stats.append((c, round(float(m.max()),3), area, xspan, yspan))
        if n_active > n_gt:
            print(os.path.basename(p), "GT=", n_gt, "active=", n_active)
            for s in stats:
                print("   chan", s)
    return 0

if __name__ == "__main__":
    sys.exit(main())
