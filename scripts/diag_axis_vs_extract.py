"""Check whether lqs whole-chain offsets come from axis mapping or extraction.

For each val image with a >2% failure, compare:
- GT data points -> pixels through the pipeline axes (value_to_pixel)
- vs meta 'curves_px' (ground-truth rendered pixel positions)
If pipeline axes map GT data correctly to GT pixels, the axis is fine and
the error is extraction-side.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np

from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.pipeline.extractor import load_config
from mci.utils import read_image

# candidate images with whole-chain offsets (from diagnosis)
STEMS = [
    "data/val_multi/img_0055",  # 29px whole-chain shift (lqs)
    "data/val_multi/img_0051",  # lqs
    "data/val_multi/img_0047",  # lqs heavy
    "data/val_multi/img_0069",  # marker_rich crossing
    "data/val_multi/img_0021",  # decelerating right-end loss (heaviest)
]

for stem in STEMS:
    if not os.path.exists(stem + ".png"):
        continue
    meta = json.load(open(stem + "_meta.json", encoding="utf-8"))
    cj = json.load(open(stem + "_curves.json", encoding="utf-8"))
    img = read_image(stem + ".png")
    cfg = load_config()
    structure = detect_structure(img, cfg)
    ocr = StubOCRBackend(stem + "_labels.json")
    x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
    x_axis, y_axis = build_axes(
        x_ticks, y_ticks,
        x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
        y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
    )
    print(f"\n== {os.path.basename(stem)} deg={meta.get('degradations')} axes={meta['x_kind']}/{meta['y_kind']}")
    # GT rendered pixels (curves_px) vs pipeline-mapped GT data
    for ci, c in enumerate(cj["curves"]):
        csv_path = os.path.join(os.path.dirname(stem), c["csv"])
        if not os.path.exists(csv_path):
            continue
        rows = [r for r in open(csv_path, encoding="utf-8").readlines() if r.strip() and not r.startswith("#")]
        data = np.array([[float(r.split(",")[0]), float(r.split(",")[1])] for r in rows[1:]])
        # pipeline pixel positions
        gx = x_axis.value_to_pixel(np.where(data[:, 0] > 0, data[:, 0], 1e-9))
        gy = y_axis.value_to_pixel(np.where(data[:, 1] > 0, data[:, 1], 1e-9))
        # meta curves_px (rendered truth, y-down image coords)
        cpx = np.asarray(meta["curves_px"][ci], dtype=np.float64) if ci < len(meta["curves_px"]) else None
        if cpx is None or len(cpx) < 2:
            continue
        # curves_px is every 10th rendered point: interpolate pipeline-mapped
        # (gx, gy) at the curves_px x positions for a like-for-like compare
        o = np.argsort(gx)
        gx_s, gy_s = gx[o], gy[o]
        inside = (cpx[:, 0] >= gx_s[0]) & (cpx[:, 0] <= gx_s[-1])
        if inside.sum() < 2:
            continue
        gy_i = np.interp(cpx[inside, 0], gx_s, gy_s)
        d = np.abs(gy_i - cpx[inside, 1])
        dx = np.abs(cpx[inside, 0] - cpx[inside, 0])  # x is the same reference
        print(f"  C{ci+1}: pipeline-vs-truth pixel dev: dy med={np.median(d):.2f} mean={d.mean():.2f} "
              f"max={d.max():.2f}")
