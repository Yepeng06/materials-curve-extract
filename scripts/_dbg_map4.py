"""Debug v4: buffer_rgba — definitionally consistent pixel space."""
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

buf = np.asarray(fig.canvas.buffer_rgba())  # HxWx4, RGBA
print("buffer:", buf.shape)
gray = buf[:, :, :3].mean(axis=2)
ink = gray < 230

disp = ax.transData.transform((50.0, 3.896))
px, py = int(disp[0]), int(disp[1])  # display x -> buffer col; display y up -> buffer row = H - y
py = buf.shape[0] - py
print("curve pred at:", px, py)
patch = gray[py - 3 : py + 4, px - 3 : px + 4]
print("patch min:", patch.min())

# ink bbox
ys, xs = np.nonzero(ink)
print("ink bbox x:", xs.min(), xs.max(), " y:", ys.min(), ys.max())
for xv in (0, 50, 100):
    yv = 1 + 3 * (1 - np.exp(-xv / 50)) + 0.02 * xv
    dx, dy = ax.transData.transform((xv, yv))
    px, py = int(dx), buf.shape[0] - int(dy)
    patch = gray[py - 2 : py + 3, px - 2 : px + 3]
    print(f"data ({xv},{yv:.2f}) -> px ({px},{py})  patch min={patch.min()}")
