"""Debug the display->saved-image pixel mapping of gen_synthetic."""
import io
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

dpi = 200
fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=dpi)
t = np.linspace(0, 100, 400)
y = 1 + 3 * (1 - np.exp(-t / 50)) + 0.02 * t
ax.plot(t, y, color="#1f77b4", lw=2)
ax.set_xlim(0, 100)
ax.set_ylim(0, 8)
fig.canvas.draw()

# probe display coords of a data point
pt = (50.0, 4.0)
disp = ax.transData.transform(pt)
print("display (y up):", disp)
print("fig size inches:", fig.get_size_inches(), "dpi:", fig.dpi)
fig_h = fig.get_size_inches()[1] * dpi
print("fig_h px:", fig_h)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
print("saved image:", img.shape)
bb = fig.bbox_inches
print("bbox_inches:", bb)

sx = img.shape[1] / (bb.width * dpi)
sy = img.shape[0] / (bb.height * dpi)
ox = bb.x0 * dpi
oy = bb.y0 * dpi
px = (disp[0] - ox) * sx
py = (fig_h - disp[1] - oy) * sy
print("mapped pixel:", px, py)

# dark pixel expected at (50,4) on the curve
patch = img[int(py) - 3:int(py) + 4, int(px) - 3:int(px) + 4]
print("patch min/max luminance:", patch.min(), patch.max())

# also try the direct renderer approach: renderer bbox
renderer = fig.canvas.get_renderer()
print("renderer canvas size:", renderer.get_canvas_width_height())
