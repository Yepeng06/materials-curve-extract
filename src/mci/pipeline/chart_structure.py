"""Chart structure detection — classical-CV baseline.

Finds, for a single-plot chart:
  * the bottom (x) axis line row,
  * the left (y) axis line column,
  * a plot bounding box that fully contains the curve pixels,
  * the pixel positions of the tick marks along both axes.

Important design note: the bounding box only needs to *contain* the curve —
data mapping is derived purely from tick positions, so bbox precision does
not affect accuracy.  A too-small bbox clips the curve; a slightly too-large
bbox is handled downstream by component scoring.

This class is the future replacement point for the YOLOv8-nano detector:
the YOLO version will return the same ``ChartStructure`` dataclass.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..schema import ChartStructure, StructureDetectionError
from ..utils import ink_mask


def _group_runs(values: np.ndarray, gap: int = 3) -> List[float]:
    """Group sorted indices into runs; return run centers."""
    if len(values) == 0:
        return []
    centers: List[float] = []
    run = [values[0]]
    for v in values[1:]:
        if v - run[-1] <= gap:
            run.append(v)
        else:
            centers.append(float(np.mean(run)))
            run = [v]
    centers.append(float(np.mean(run)))
    return centers


def detect_structure(image_bgr: np.ndarray, cfg: Dict | None = None) -> ChartStructure:
    cfg = cfg or {}
    frac_thr = float(cfg.get("axis_frac_threshold", 0.45))
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray)

    # ---- horizontal profile over the central 92% width -> bottom axis ----
    cx0, cx1 = int(w * 0.04), int(w * 0.96)
    row_frac = ink[:, cx0:cx1].mean(axis=1).astype(np.float64)
    rows = np.where(row_frac > frac_thr)[0]
    if len(rows) == 0:
        raise StructureDetectionError("no horizontal axis line found")
    y_axis = int(rows[-1])

    # ---- vertical profile above the axis -> left axis ----
    col_frac = ink[: y_axis + 1, :].mean(axis=0).astype(np.float64)
    cols = np.where(col_frac > frac_thr)[0]
    if len(cols) == 0:
        raise StructureDetectionError("no vertical axis line found")
    x_axis = int(cols[0])

    # ---- content bounds inside the plot ----
    band_x0 = x_axis + 2
    band_x1 = w - 1
    region = ink[: y_axis, band_x0:band_x1]
    if region.size == 0 or not region.any():
        raise StructureDetectionError("no content found inside the plot area")
    col_any = region.any(axis=0)
    row_any = region.any(axis=1)
    y0 = int(np.argmax(row_any))  # topmost ink row
    x1 = int(np.argmax(col_any[::-1]))  # rightmost ink col (within band)
    x1 = band_x1 - 1 - x1
    if x1 < band_x0:
        x1 = band_x0

    # Trim columns whose ink only appears in the top 25% of the band
    # (e.g. a legend anchored outside the plot, top-right).
    band_h = max(1, y_axis - y0)
    band_low = int(y0 + band_h * 0.75)
    trimmed = 0
    while x1 > band_x0:
        col = ink[y0 : y_axis + 1, x1]
        if col.any() and not col[band_low - y0 :].any() and trimmed < max(30, w // 10):
            x1 -= 1
            trimmed += 1
        else:
            break
    x1 = max(x1, band_x0)
    y0 = max(y0, 0)
    if y_axis - y0 < 10 or x1 - band_x0 < 10:
        raise StructureDetectionError("plot area too small after detection")

    # ---- tick marks ----
    # Ticks are short strokes attached to the axis line; tick labels sit
    # further out with a gap.  Detect ink in the narrow band right below /
    # left of the axis so label glyph tops are never mistaken for ticks,
    # and reject columns whose ink continues into the plot interior (the
    # curve hugging the bottom axis would otherwise pollute the ticks).
    tick_band = 5
    # x-axis ticks: short vertical strokes just below the bottom axis line,
    # restricted to the right of the y-axis (y tick labels live left of it
    # and would otherwise pollute the band at the same rows)
    x_band = ink[y_axis + 2 : min(y_axis + 2 + tick_band, h), x_axis + 2 :]
    x_ticks_px = []
    for col in np.where(x_band.sum(axis=0) >= 2)[0]:
        col = x_axis + 2 + col
        if ink[max(0, y_axis - 12) : y_axis - 2, col].any():
            continue  # connected to interior content (the curve), not a tick
        x_ticks_px.append(float(col))
    x_ticks_px = _group_runs(np.asarray(x_ticks_px))
    # y-axis ticks: short horizontal strokes just left of the left axis line
    # (the curve is always to the RIGHT of the axis, so no interior check)
    y_band = ink[:, max(0, x_axis - 2 - tick_band) : max(0, x_axis - 2)]
    y_ticks_px = _group_runs(np.where(y_band.sum(axis=1) >= 2)[0])

    structure = ChartStructure(
        plot_bbox=(band_x0, y0, x1, y_axis - 1),
        x_axis_pixel=y_axis,
        y_axis_pixel=x_axis,
        x_ticks_px=x_ticks_px,
        y_ticks_px=y_ticks_px,
    )
    return structure
