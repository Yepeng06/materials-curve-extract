"""Compare get_window_extent vs transData vs buffer pixels."""
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

dpi = 200
fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=dpi)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.01, 10000)
ax.set_ylim(0.01, 1000)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

buf = np.asarray(fig.canvas.buffer_rgba())
H, W = buf.shape[:2]
print("buffer:", W, H)

# tick label extents
for lb in ax.get_xticklabels():
    if not lb.get_text():
        continue
    ext = lb.get_window_extent(renderer)
    print(f"xlabel {lb.get_text()!r}: extent center=({ext.x0 + ext.width/2:.1f}, {ext.y0 + ext.height/2:.1f}) "
          f"size=({ext.width:.1f},{ext.height:.1f})")
    break
for lb in ax.get_yticklabels():
    if not lb.get_text():
        continue
    ext = lb.get_window_extent(renderer)
    print(f"ylabel {lb.get_text()!r}: extent center=({ext.x0 + ext.width/2:.1f}, {ext.y0 + ext.height/2:.1f})")
    break

# transData of the same tick values
ylim = ax.get_ylim()
xlim = ax.get_xlim()
for v in (1.0, 10.0, 100.0):
    print(f"x tick {v}: transData=({ax.transData.transform((v, ylim[0]))[0]:.1f})")
for v in (1.0, 10.0, 100.0):
    print(f"y tick {v}: transData=({ax.transData.transform((xlim[0], v))[1]:.1f})")

print("axes bbox display:", ax.get_position().transformed(fig.transFigure))
print("axes bbox pixels:", ax.get_position().transformed(fig.transFigure).get_points() * np.array([W, H]))
