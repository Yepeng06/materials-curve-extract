import sys, os
sys.path.insert(0, 'src')
import numpy as np
import cv2
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.extractor import load_config
from mci.utils import read_image, ink_mask

cfg = load_config()
path = 'data/real_diag/pmc_oa/raw/PMC3556318/1471-2458-13-41-1.jpg'
img = read_image(path)
if img.ndim == 3 and img.shape[2] == 4:
    a = img[:, :, 3] / 255.0
    img = (img[:, :, :3] * a[..., None] + 255 * (1 - a[..., None])).astype(np.uint8)
structure = detect_structure(img, cfg)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ink = ink_mask(gray)
h, w = ink.shape
x_axis = int(structure.y_axis_pixel)  # left axis column
print('y_axis_pixel (left axis col):', x_axis, 'image w:', w)

# inspect ink in the region left of the axis: columns x_axis-30 .. x_axis+5, all rows
band = ink[:, max(0, x_axis - 30): x_axis + 5]
rows_with_ink = np.nonzero(band.sum(axis=1) >= 1)[0]
print('rows with any ink left-of-axis:', len(rows_with_ink))
# per-row ink extent (columns) to see stroke lengths
import collections
lengths = collections.Counter()
for r in rows_with_ink:
    cols = np.nonzero(band[r])[0]
    n = int(cols.max() - cols.min() + 1)
    lengths[n] += 1
print('stroke length histogram (px):', dict(sorted(lengths.items())))
# sample some rows with long strokes
for r in rows_with_ink[:20]:
    cols = np.nonzero(band[r])[0]
    if cols.max() - cols.min() + 1 >= 3:
        print('row', r, 'ink cols (rel to x_axis-30):', cols.min(), '-', cols.max())
