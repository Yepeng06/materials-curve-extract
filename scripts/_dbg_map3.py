"""Debug v3: where are the dark pixels vs transData predictions."""
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

print("axes position (fraction):", ax.get_position())
for pt in [(0, 0), (100, 8), (50, 4), (0, 8)]:
    print(pt, "->", ax.transData.transform(pt))

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=dpi)
img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ink = gray < 230
print("saved:", img.shape, "ink pixels:", int(ink.sum()))
if ink.any():
    ys, xs = np.nonzero(ink)
    print("ink bbox x:", xs.min(), xs.max(), "y:", ys.min(), ys.max())
    # find ink at row 376 (expected curve y for x=50)
    row = ink[376]
    cols = np.nonzero(row)[0]
    print("row 376 ink cols:", cols[:20], "..." if len(cols) > 20 else "")
    # column 563
    col = ink[:, 563]
    rows = np.nonzero(col)[0]
    print("col 563 ink rows:", rows[:20], "..." if len(rows) > 20 else "")
