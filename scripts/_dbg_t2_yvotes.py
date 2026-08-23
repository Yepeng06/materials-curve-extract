import sys
sys.path.insert(0, 'src')
import numpy as np
import cv2
from mci.pipeline.chart_structure import detect_structure, _group_runs
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
x_axis = int(structure.y_axis_pixel)
y_axis = int(structure.x_axis_pixel)
axis_top = structure.plot_bbox[1]
print('x_axis(col)=', x_axis, 'y_axis(row)=', y_axis, 'axis_top=', axis_top)

tick_scales = (2, 5, 8)
y_votes = {}
for bw in tick_scales:
    lo = max(0, x_axis - 2 - bw)
    hi = max(0, x_axis - 2)
    y_band = ink[:, lo:hi]
    rows = np.where(y_band.sum(axis=1) >= 2)[0]
    n = 0
    for row in rows:
        row = int(row)
        if axis_top <= row <= y_axis:
            continue
        y_votes[row] = y_votes.get(row, 0) + 1
        n += 1
    print(f'bw={bw} cols[{lo}:{hi}] rows_ge2={n}')
kept = [r for r, n in y_votes.items() if n >= 2]
print('kept rows (>=2 votes):', len(kept), kept[:20])
print('grouped:', _group_runs(np.asarray(kept, dtype=np.float64)))

# where are the 5px strokes?
for r in range(0, 360, 10):
    cols = np.nonzero(ink[r, max(0, x_axis - 15): x_axis + 10])[0]
    if len(cols):
        print('row', r, 'cols rel x_axis-15:', cols.min(), '-', cols.max(), 'len', cols.max() - cols.min() + 1)
