"""Bar-chart heuristic: detect rectangular bars inside the plot region.

Non-target charts (bar charts) currently surface as AxisFitError in the
zero-shot real-image diagnosis, inflating the 'failure' count.  This
module flags plots whose ink is dominated by rectangular bars so the
evaluator can classify them as non-target instead of failures.
"""
from __future__ import annotations

import cv2
import numpy as np

from .chart_structure import ChartStructure
from ..utils import ink_mask


def is_bar_chart(image_bgr: np.ndarray, structure: ChartStructure,
                 min_bars: int = 3) -> bool:
    """True when the plot region contains >= min_bars rectangle-like bars.

    Heuristic: connected components inside the plot that are mostly solid
    (fill ratio > 0.6 of their bbox), reasonably large (>= 120 px), and
    with a strongly elongated aspect ratio (>= 2.5) OR a wide flat band
    (vertical bars) / tall narrow band (horizontal bars).  Curves are thin
    (fill ratio ~0.2-0.4), so they do not qualify.
    """
    x0, y0, x1, y1 = structure.plot_bbox
    crop = image_bgr[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0 or min(crop.shape[:2]) < 20:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    bars = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 120 or w < 3 or h < 3:
            continue
        fill = area / float(w * h)
        if fill < 0.6:
            continue  # thin curve, not a bar
        aspect = max(w, h) / float(max(min(w, h), 1))
        if aspect >= 2.5 or area >= 1500:
            bars += 1
    return bars >= min_bars
