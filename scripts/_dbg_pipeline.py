"""Step-by-step pipeline debug on one image."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from mci.pipeline.chart_structure import detect_structure  # noqa: E402
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks  # noqa: E402
from mci.utils import read_image  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic/img_0000.png"
img = read_image(path)
print("image:", path, img.shape)

meta_path = os.path.splitext(path)[0] + "_meta.json"
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("GT:", meta["x_kind"], meta["y_kind"], "x_range", meta["x_range"], "y_range", meta["y_range"])
    print("GT x_tick_values:", meta.get("x_tick_values"))
    print("GT y_tick_values:", meta.get("y_tick_values"))

s = detect_structure(img)
print("plot_bbox:", s.plot_bbox, "x_axis_pixel:", s.x_axis_pixel, "y_axis_pixel:", s.y_axis_pixel)
print("x_ticks_px:", s.x_ticks_px)
print("y_ticks_px:", s.y_ticks_px)

labels_path = os.path.splitext(path)[0] + "_labels.json"
with open(labels_path, "r", encoding="utf-8") as f:
    labels = json.load(f)
print("\nstub labels:")
for l in labels:
    print("  ", l["text"], l["box"], "center:", np.mean(l["box"], axis=0))

ocr = StubOCRBackend(labels_path)
x_ticks, y_ticks = read_ticks(img, s, ocr)
print("\nx_ticks:")
for t in x_ticks:
    print("   pixel=%.1f value=%s text=%r score=%.2f" % (t.pixel, t.value, t.text, t.score))
print("y_ticks:")
for t in y_ticks:
    print("   pixel=%.1f value=%s text=%r score=%.2f" % (t.pixel, t.value, t.text, t.score))

from mci.pipeline.coordinate_mapper import build_axes  # noqa: E402

xa, ya = build_axes(x_ticks, y_ticks)
print("\naxes: x=%s (q=%.4f, %d ticks)  y=%s (q=%.4f, %d ticks)" % (
    xa.kind.value, xa.quality, sum(1 for t in x_ticks if t.value is not None),
    ya.kind.value, ya.quality, sum(1 for t in y_ticks if t.value is not None)))

from mci.pipeline.curve_extractor import extract_curves  # noqa: E402

curves = extract_curves(img, s, xa, ya)
print("curve points:", len(curves[0].points))

