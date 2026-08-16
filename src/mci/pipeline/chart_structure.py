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
    edge_frac = float(cfg.get("axis_edge_exclude", 0.03))
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray)

    # ---- horizontal profile over the central 92% width -> bottom axis ----
    cx0, cx1 = int(w * 0.04), int(w * 0.96)
    row_frac = ink[:, cx0:cx1].mean(axis=1).astype(np.float64)
    rows = np.where(row_frac > frac_thr)[0]
    if len(rows) == 0:
        raise StructureDetectionError("no horizontal axis line found")
    # Robustness: a full-width figure border / crop residue below the axis
    # must not be picked as the bottom axis line, so exclude bottom-edge
    # rows first (fall back to the raw rows if the cut removes everything).
    bottom_cut = int(h * (1.0 - edge_frac))
    rows_ok = rows[rows < bottom_cut]
    if len(rows_ok) == 0:
        rows_ok = rows
    y_axis = int(rows_ok[-1])
    # x-axis line thickness: the contiguous run of ink-full rows ending at
    # y_axis (lines render 1-5 px depending on width/antialiasing).  Used to
    # keep the axis line itself out of tick/interior checks below.
    axis_top = y_axis
    for r in rows_ok[::-1][1:]:
        if r == axis_top - 1:
            axis_top = r
        else:
            break

    # ---- vertical profile above the axis -> left axis ----
    # Central scan: exclude the left/right edge columns (failure sample
    # fail_001 had left-edge crop residue misdetected as the y axis, which
    # set y_axis_pixel=0 and collapsed the OCR strip).  Connectivity check:
    # the y-axis line must reach the x-axis line at the corner (ink in the
    # rows around y_axis); otherwise try the next candidate column.
    edge_w = max(3, int(w * edge_frac))
    col_frac = ink[: y_axis + 1, edge_w : w - edge_w].mean(axis=0).astype(np.float64)
    cols = np.where(col_frac > frac_thr)[0] + edge_w
    if len(cols) == 0:
        raise StructureDetectionError("no vertical axis line found")
    x_axis = int(cols[0])
    if not ink[max(0, y_axis - 3) : y_axis + 3, x_axis].any():
        for c in cols[1:]:
            if ink[max(0, y_axis - 3) : y_axis + 3, int(c)].any():
                x_axis = int(c)
                break

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

    # ---- tick marks (multi-scale band voting) ----
    # Ticks are short strokes attached to the axis line; tick labels sit
    # further out with a gap.  A single fixed band width is brittle (long
    # ticks fall out of a narrow band; label glyphs pollute a wide one), so
    # detect across several band widths and keep positions seen by >= 2
    # scales.  Columns whose ink continues into the plot interior (the curve
    # hugging the bottom axis) are still rejected.
    tick_scales = (2, 5, 8)
    # interior connectivity check band: strictly above the axis line itself
    interior_top = max(0, axis_top - 12)
    interior_bot = max(0, axis_top - 2)
    # x-axis ticks: short vertical strokes just below the bottom axis line,
    # restricted to the right of the y-axis (y tick labels live left of it
    # and would otherwise pollute the band at the same rows)
    x_votes: Dict[int, int] = {}
    for bw in tick_scales:
        x_band = ink[y_axis + 2 : min(y_axis + 2 + bw, h), x_axis + 2 :]
        for col in np.where(x_band.sum(axis=0) >= 2)[0]:
            col = x_axis + 2 + int(col)
            if ink[interior_top:interior_bot, col].any():
                continue  # connected to interior content (the curve), not a tick
            x_votes[col] = x_votes.get(col, 0) + 1
    x_ticks_px = _group_runs(
        np.asarray([c for c, n in x_votes.items() if n >= 2], dtype=np.float64)
    )
    # y-axis ticks: short horizontal strokes just left of the left axis line
    # (the curve is always to the RIGHT of the axis, so no interior check;
    # the x-axis line itself is excluded by row range)
    y_votes: Dict[int, int] = {}
    for bw in tick_scales:
        y_band = ink[:, max(0, x_axis - 2 - bw) : max(0, x_axis - 2)]
        for row in np.where(y_band.sum(axis=1) >= 2)[0]:
            row = int(row)
            if axis_top <= row <= y_axis:
                continue  # the x-axis line itself, not a y tick
            y_votes[row] = y_votes.get(row, 0) + 1
    y_ticks_px = _group_runs(
        np.asarray([r for r, n in y_votes.items() if n >= 2], dtype=np.float64)
    )

    structure = ChartStructure(
        plot_bbox=(band_x0, y0, x1, y_axis - 1),
        x_axis_pixel=y_axis,
        y_axis_pixel=x_axis,
        x_ticks_px=x_ticks_px,
        y_ticks_px=y_ticks_px,
    )
    return structure
