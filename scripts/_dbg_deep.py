"""Deep debug: components, scores, trace for one image."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np

from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import _column_median, _remove_gridlines, _trace_chain
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import ink_mask, read_image

path = sys.argv[1]
img = read_image(path)
meta = json.load(open(os.path.splitext(path)[0] + "_meta.json"))
print("GT kinds:", meta["x_kind"], meta["y_kind"], "ranges:", meta["x_range"], meta["y_range"])
print("degrad:", meta["degradations"])

s = detect_structure(img)
print("bbox:", s.plot_bbox, "x_axis:", s.x_axis_pixel, "y_axis:", s.y_axis_pixel)
x_ticks, y_ticks = read_ticks(img, s, StubOCRBackend(os.path.splitext(path)[0] + "_labels.json"))
xa, ya = build_axes(x_ticks, y_ticks)
print("axes: x=%s y=%s (q %.4f/%.4f)" % (xa.kind.value, ya.kind.value, xa.quality, ya.quality))

x0, y0, x1, y1 = s.plot_bbox
crop = img[y0 : y1 + 1, x0 : x1 + 1]
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
ink = ink_mask(gray)
ink[:3, :] = 0; ink[-3:, :] = 0; ink[:, :3] = 0; ink[:, -3:] = 0
print("ink frac (before grid removal): %.4f" % ink.mean())

cleaned = _remove_gridlines(ink, img, s, {})
print("ink frac (after grid removal):  %.4f" % cleaned.mean())

n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
comps = []
rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 60:
        continue
    edge_h = (y <= 6) or (y + h >= plot_h - 6)
    edge_w = (x <= 6) or (x + w >= plot_w - 6)
    if edge_h and edge_w and (h <= 6 or w <= 6):
        continue
    px = rgb[labels == i]
    mg = float(np.mean(px, axis=1).mean()) if len(px) else 255.0
    dark = float(np.clip((255.0 - mg) / 128.0, 0.3, 2.0))
    span = w / plot_w
    comps.append((span * np.sqrt(area) * dark, i, area, span, (x, y, w, h), round(mg, 1)))
comps.sort(key=lambda c: -c[0])
print("top components (score, id, area, span, bbox, gray):")
for c in comps[:6]:
    print("   ", c)

idx = comps[0][1]
mask = (labels == idx).astype(np.uint8)
print("selected component area:", int(mask.sum()), "bbox span:", comps[0][4])

from skimage.morphology import skeletonize

skel = skeletonize(mask.astype(bool)).astype(np.uint8)
chain = _trace_chain(skel)
if chain:
    xs = [p[0] for p in chain]
    ys = [p[1] for p in chain]
    print("trace: len=%d x[%d..%d] y[%d..%d]" % (len(chain), min(xs), max(xs), min(ys), max(ys)))
else:
    print("trace: None -> column median")
    cm = _column_median(mask)
    print("column median len:", len(cm))
