"""Debug v2: check exact 1:1 mapping with bbox_inches=None."""
import io

import cv2
import numpy as np

import matplotlib

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
disp = ax.transData.transform((50.0, 4.0))
print("display:", disp)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=dpi)  # no tight bbox
img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
print("saved:", img.shape)

px, py = int(disp[0]), int(disp[1])
print("curve at pixel:", px, py)
patch = img[py - 3 : py + 4, px - 3 : px + 4]
print("patch min/max:", patch.min(), patch.max())

# sample along the curve: x=25,50,75 -> y = f(x)
for xv in (25, 50, 75):
    yv = 1 + 3 * (1 - np.exp(-xv / 50)) + 0.02 * xv
    dx, dy = ax.transData.transform((xv, yv))
    px, py = int(dx), int(dy)
    patch = img[py - 3 : py + 4, px - 3 : px + 4]
    print(f"({xv},{yv:.2f}) -> px {px},{py}  min={patch.min()}")
