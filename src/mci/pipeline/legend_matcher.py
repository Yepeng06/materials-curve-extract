"""Legend matching — associate legend entries with curves (multi-curve).

Baseline was a no-op (single curve).  Phase C multi-curve + real charts:
the legend box text (OCR boxes) and curve colours (Curve.color, already
computed from the mask pixels) are matched by colour distance with a
greedy assignment, filling Curve.legend_label.

Design notes (research review 2026-08-19):
  * LineEX showed legend-dependent extraction fails without a legend, so
    this stage is an *annotation* stage: curves are never dropped when the
    legend is missing, only labelled when it is found.
  * ChartZero uses a VLM for legend binding (F1 0.945) — we use colour
    distance as the CPU-only fallback; a VLM pass is a later option.
  * Synthetic platform charts are greyscale (BW palette) so colour matching
    is weak there; real paper charts have coloured curves where it shines.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schema import ChartStructure, Curve
from .base import TextBox


def _legend_text_boxes(
    text_boxes: List[TextBox], structure: ChartStructure
) -> List[TextBox]:
    """Candidate legend entries: short text boxes inside the plot area
    (corner legends) or in the right margin (outside legends).

    Filter: keep boxes with 1-40 characters that are NOT purely numeric
    (tick labels) — legend entries typically sit in the upper right /
    upper left / right margin of the plot bbox."""
    x0, y0, x1, y1 = structure.plot_bbox
    margin = max(10, int((x1 - x0) * 0.15))
    out = []
    for b in text_boxes:
        cx, cy = b.center
        t = b.text.strip()
        if not (1 <= len(t) <= 40):
            continue
        letters = sum(c.isalpha() for c in t)
        if letters == 0:
            continue
        inside = (x0 <= cx <= x1 and y0 <= cy <= y1)
        right_margin = (x1 < cx <= x1 + margin and y0 <= cy <= y1)
        if inside or right_margin:
            out.append(b)
    return out


def _colour_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """RGB distance (curve colours are BGR tuples in this codebase)."""
    a = np.asarray(c1[::-1], dtype=np.float64)  # BGR -> RGB
    b = np.asarray(c2[::-1], dtype=np.float64)
    return float(np.linalg.norm(a - b))


def _sample_legend_colour(
    image_bgr: np.ndarray, box: TextBox, pad: int = 3
) -> Optional[Tuple[int, int, int]]:
    """Median colour of the non-white pixels just left of the legend text
    (matplotlib legend line-sample swatch sits before the text)."""
    try:
        import cv2
    except Exception:
        return None
    b = box.box.astype(np.float64)
    x0, y0 = b.min(axis=0).astype(int)
    x1, y1 = b.max(axis=0).astype(int)
    h, w = image_bgr.shape[:2]
    sw_x0 = max(0, x0 - 14 - pad)
    sw_x1 = max(0, x0 - pad)
    if sw_x1 - sw_x0 < 4:
        return None
    region = image_bgr[max(0, y0 - pad): min(h, y1 + pad), sw_x0:sw_x1].reshape(-1, 3)
    if len(region) < 8:
        return None
    # keep non-white ink only (median of the swatch line colour, not the paper)
    dark = region[(region < 235).any(axis=1)]
    if len(dark) < 4:
        return None
    return tuple(int(v) for v in np.median(dark, axis=0))


def match_legends(
    curves: List[Curve],
    structure: ChartStructure,
    text_boxes: List[TextBox],
    cfg: Optional[Dict] = None,
    image_bgr: Optional[np.ndarray] = None,
) -> List[Curve]:
    """Label curves from legend entries by colour similarity.

    No legend boxes -> identity (single-curve / no-legend charts).  When
    legend entries are found, each curve is assigned the entry whose colour
    is nearest (greedy, one entry per curve); curves keep their geometry
    either way — labels are annotations, not filters."""
    cfg = cfg or {}
    if len(curves) <= 1:
        return curves
    entries = _legend_text_boxes(text_boxes, structure)
    if not entries:
        return curves
    thr = float(cfg.get("legend_colour_thr", 120.0))
    entry_colours = []
    for e in entries:
        c = None
        if image_bgr is not None:
            c = _sample_legend_colour(image_bgr, e)
        entry_colours.append((e, c))
    used = set()
    for curve in curves:
        if curve.color is None:
            continue
        best = None
        for i, (entry, ec) in enumerate(entry_colours):
            if i in used or ec is None:
                continue
            d = _colour_distance(curve.color, ec)
            if best is None or d < best[0]:
                best = (d, i, entry.text)
        if best is not None and best[0] <= thr:
            used.add(best[1])
            curve.legend_label = best[2]
    return curves