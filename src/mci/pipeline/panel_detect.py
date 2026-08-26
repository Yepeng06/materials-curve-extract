"""Multi-panel (subplot) detection for real-world chart images.

Algorithm (validated on the 326-image creep-curve library, see
``dataset/panel_estimate.csv``): Otsu binarization -> morphological closing ->
closed 4-corner contour detection.  Grid-immune by construction: grid lines do
not form closed rectangles, curves rarely enclose rectangles >= 3% of the
image area.

Conservative by design: only large, well-formed rectangles are reported and
nested frames are collapsed to the inner one.  False positives are acceptable
for *explainable reject* purposes (they push a failure into quality-B with a
hint) but must never break the successful single-panel path — callers should
only consult ``detect_panels`` when the main pipeline failed or when the
quality gate is explicitly enabled.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]  # (x0, y0, x1, y1) inclusive


def detect_panels(
    image_bgr: np.ndarray,
    min_area_frac: float = 0.03,
    max_area_frac: float = 0.97,
    min_side_frac: float = 0.10,
) -> List[Rect]:
    """Return axis-frame rectangles (x0, y0, x1, y1) found in the image.

    ``image_bgr``: BGR image as produced by ``read_image``.
    Returns an empty list when no panel is detected (single-panel layout with
    an unclosed frame, dark backgrounds, photos, etc.).
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # close small gaps in axis frames (dashed frames, thin lines)
    kernel = np.ones((5, 5), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects: List[Tuple[int, int, int, int, float]] = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4:
            continue
        x, y, ww, hh = cv2.boundingRect(approx)
        area = float(ww) * hh
        if area < min_area_frac * w * h or area > max_area_frac * w * h:
            continue
        if ww < min_side_frac * w or hh < min_side_frac * h:
            continue
        rects.append((x, y, x + ww - 1, y + hh - 1, area))

    # step 1: merge near-duplicate contours (inner + outer edges of the same
    # frame produce two almost-coincident rectangles after closing); only when
    # areas are comparable (ratio < 1.3) — genuinely nested frames (area
    # ratio >> 1.3) are left for step 2 to resolve.
    merged: List[Tuple[int, int, int, int, float]] = []
    for r in rects:
        x0, y0, x1, y1, a = r
        hit = None
        for i, m in enumerate(merged):
            mx0, my0, mx1, my1, ma = m
            contains = (mx0 - 3 <= x0 and my0 - 3 <= y0 and x1 <= mx1 + 3
                        and y1 <= my1 + 3)
            inside = (x0 - 3 <= mx0 and y0 - 3 <= my0 and mx1 <= x1 + 3
                      and my1 <= y1 + 3)
            ratio = max(a, ma) / min(a, ma) if min(a, ma) > 0 else 1.0
            if (contains or inside) and ratio < 1.3:
                hit = i
                break
        if hit is None:
            merged.append(r)
        elif a > merged[hit][4]:
            merged[hit] = r

    # step 2: keep the innermost frame of each nesting chain (drop outer
    # frames that contain a significantly smaller inner frame — e.g. a
    # page/decorative border around the actual plot frame).
    kept: List[Rect] = []
    for r in merged:
        x0, y0, x1, y1, a = r
        has_inner = False
        for q in merged:
            if q is r:
                continue
            qx0, qy0, qx1, qy1, qa = q
            if (x0 <= qx0 and y0 <= qy0 and qx1 <= x1 and qy1 <= y1
                    and a > qa * 1.3):
                has_inner = True
                break
        if not has_inner:
            kept.append((x0, y0, x1, y1))
    # deterministic order: left-to-right, then top-to-bottom
    kept.sort(key=lambda r: (r[0], r[1]))
    return kept


def panel_grid_shape(panels: List[Rect], tol_frac: float = 0.08) -> Optional[Tuple[int, int]]:
    """Infer (rows, cols) grid layout of panels; None if irregular."""
    if not panels:
        return None
    xs = sorted({p[0] for p in panels})
    ys = sorted({p[1] for p in panels})
    # group near-identical origins
    def cluster(vals):
        groups: List[List[float]] = []
        for v in sorted(vals):
            if groups and abs(v - groups[-1][-1]) <= max(8.0, tol_frac * 100.0):
                groups[-1].append(v)
            else:
                groups.append([v])
        return groups
    cols = len(cluster(xs))
    rows = len(cluster(ys))
    if rows * cols != len(panels):
        return None  # irregular layout (mixed panels), treat as unknown
    return (rows, cols)
