"""Export-layer point resampling (user-selectable point count).

The extraction pipeline's native density (``max_points=2000`` pixel chain)
stays untouched — per goal.md §任务0.1 the export point density is DECOUPLED
from the evaluation protocol.  Resampling happens only here, at export time,
in DATA coordinate space:

* x-uniform interpolation over the curve's own x-range (x ascending);
* on a log x-axis the interpolation runs in log10(x) space so sampled
  points land uniformly per decade (industry habit, cf. WebPlotDigitizer
  "N points" export);
* y is always interpolated linearly in data space (log y stays a value).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _finite(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    out = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            out.append((x, y))
    return out


def resample_points(
    points: Sequence[Sequence[float]],
    n: int,
    x_log: bool = False,
) -> List[Tuple[float, float]]:
    """Resample a curve to ``n`` points (x-uniform, data space).

    Parameters
    ----------
    points
        Curve points ``[(x, y), ...]`` in data coordinates (any order).
    n
        Target point count.  ``n <= 0`` returns the input unchanged
        ("native" density option).
    x_log
        When True the x-axis is logarithmic: sample uniformly in log10(x).

    Returns
    -------
    list of (x, y)
        ``n`` points, x ascending.  Fewer than ``n`` when the curve has
        fewer than 2 finite points or a degenerate x-range.
    """
    pts = _finite(points)
    if n is None or n <= 0 or len(pts) < 2:
        return sorted(pts, key=lambda p: p[0])

    pts.sort(key=lambda p: p[0])
    xs = np.asarray([p[0] for p in pts], dtype=float)
    ys = np.asarray([p[1] for p in pts], dtype=float)

    # Strictly increasing sample axis: log10(x) for log axes, x otherwise.
    # Duplicate x (vertical segments) collapse to the mean y before interp.
    if x_log:
        if xs.min() <= 0:
            return pts  # log domain violated; keep native points
        sx = np.log10(xs)
    else:
        sx = xs
    keep = np.concatenate([[True], np.diff(sx) > 0])
    if keep.sum() < 2:
        return pts
    sx_u, ys_u = sx[keep], ys[keep]
    if x_log:
        # average y over duplicate x before collapsing (keep mask drops dup)
        ys_u = np.asarray([
            ys[(sx == v)].mean() for v in sx_u
        ], dtype=float)

    grid = np.linspace(sx_u[0], sx_u[-1], int(n))
    y_out = np.interp(grid, sx_u, ys_u)
    x_out = 10.0 ** grid if x_log else grid
    return [(float(a), float(b)) for a, b in zip(x_out, y_out)]


def resample_result_curves(curves: List, n: int, x_log: bool = False) -> List:
    """Return shallow-copied ``Curve`` objects with resampled ``points``.

    ``curves`` items are ``mci.schema.Curve``; pixel chains and colors are
    carried over unchanged so overlay/visual exports keep working.
    """
    from ..schema import Curve  # local import to avoid a cycle at module load

    out: List[Curve] = []
    for c in curves:
        pts = resample_points(c.points, n, x_log=x_log)
        out.append(Curve(
            name=c.name,
            points=pts,
            pixel_points=list(c.pixel_points),
            color=tuple(c.color) if c.color else (0, 0, 0),
            legend_label=c.legend_label,
        ))
    return out
