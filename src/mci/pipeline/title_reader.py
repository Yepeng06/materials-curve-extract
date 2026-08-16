"""Axis title / title / unit recognition (Phase B-2).

Classifies whole-image OCR text boxes into roles -- title / x axis
label / y axis label -- by position and content rules, and extracts
the variable name and unit from axis labels like 'Creep strain (%)'
or 'Time (h)'.  A 'log'/'ln' axis label produces a log prior that is
fed into fit_axis as kind_hint (B-3 signal 3).

Role rules:
  * title:  above the plot top, wide and centred
  * x label: below the bottom axis line, below the tick-label strip
  * y label: left of the plot, tall/narrow (rotated) text

Content rules:
  * 'Name (unit)' / 'Name / unit' / 'Name (unit)' -> variable + unit
  * leading log/ln -> log prior
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..schema import ChartStructure
from .base import TextBox


_UNIT_RE = re.compile(r"^\(?([^/(]*?)\s*[/\uFF0F]?\s*\(?([A-Za-z%\u00B0][A-Za-z0-9%\u00B0./^-]*)\)?$")
_LOG_RE = re.compile(r"^\s*[lL][oO][gG]\b|^\s*[lL][nN]\b")


def _parse_axis_label(text: str) -> Dict[str, str]:
    """Split 'Creep strain (%)' into variable + unit; detect log prior.

    Returns {text, variable, unit, log_hint} where missing fields are ''.
    """
    s = text.strip()
    out = {"text": s, "variable": "", "unit": "", "log_hint": ""}
    m = re.match(r"^(.+?)\s*[\uFF08(]\s*([^\uFF09)]+?)\s*[\uFF09)]\s*$", s)
    if m:
        out["variable"] = m.group(1).strip()
        out["unit"] = m.group(2).strip()
    else:
        parts = re.split(r"\s*/\s*", s)
        if len(parts) == 2 and parts[0] and parts[1]:
            out["variable"], out["unit"] = parts[0].strip(), parts[1].strip()
        elif len(parts) == 2 and not parts[0]:
            out["variable"], out["unit"] = "", parts[1].strip()
        else:
            out["variable"] = s
    if _LOG_RE.match(s) or " log" in s.lower() or " ln " in s.lower():
        out["log_hint"] = "log"
    return out


def read_titles(
    boxes: List[TextBox], structure: ChartStructure,
) -> Dict[str, dict]:
    """Classify OCR boxes into {title, x_label, y_label} roles.

    Returns a dict with the best box per role (or missing keys when no
    box matches).  Each value: {text, variable, unit, log_hint} from
    _parse_axis_label plus the box center.
    """
    x0, y0, x1, y1 = structure.plot_bbox
    x_axis_row = structure.x_axis_pixel
    plot_w = max(1, x1 - x0)
    plot_h = max(1, x_axis_row - y0)
    img_h = max((float(b.box[:, 1].max()) for b in boxes), default=1.0)

    def box_area(b: TextBox) -> float:
        xs = b.box[:, 0]
        ys = b.box[:, 1]
        return float((xs.max() - xs.min()) * (ys.max() - ys.min()))

    best: Dict[str, Tuple[float, TextBox]] = {}
    for b in boxes:
        cx, cy = b.center
        w_b = float(b.box[:, 0].max() - b.box[:, 0].min())
        h_b = float(b.box[:, 1].max() - b.box[:, 1].min())
        if h_b <= 0 or w_b <= 0:
            continue
        role = None
        # title: wide text band near the image top.  NOTE: plot_bbox.y0
        # may itself be polluted by title glyphs, so the image top is the
        # anchor, not y0.
        if cy < 0.15 * img_h and w_b > 0.25 * plot_w and h_b < 0.1 * img_h:
            role = "title"
        # x axis label: below the bottom axis line, under the tick-label
        # strip, roughly centred (legend is inside the plot, above)
        elif cy > x_axis_row + max(30, plot_h * 0.06) and h_b < 0.1 * plot_h:
            role = "x_label"
        # y axis label: left 25% of the plot, above the axis; either a
        # tall/narrow rotated box, or a wider horizontal box whose text
        # carries a unit (y tick labels are small and unit-less)
        elif (cx < x0 + 0.25 * plot_w and cy < x_axis_row
              and (h_b > 1.5 * w_b
                   or (w_b > 0.06 * plot_w and any(k in b.text for k in ("(", "/", "%", "MPa", "mm", "s", "h"))))):
            role = "y_label"
        if role is None:
            continue
        key = role
        if key not in best or box_area(b) > best[key][0]:
            best[key] = (box_area(b), b)

    out: Dict[str, dict] = {}
    for role, (_, b) in best.items():
        parsed = _parse_axis_label(b.text)
        parsed["center"] = [round(b.center[0], 1), round(b.center[1], 1)]
        out[role] = parsed
    return out


def read_rotated_y_title(
    image_bgr: np.ndarray, structure: ChartStructure, ocr: "object",
) -> Optional[dict]:
    """OCR the y-axis title band rotated 90 deg (vertical paper-style axis
    titles are unreadable in place; PP-OCRv4 handles the rotated crop well).

    Crops the left band (left 25% of the plot, above the x axis), rotates it
    counter-clockwise so the vertical text reads left-to-right, OCRs at 2x
    and keeps the largest non-numeric box.  Returns the parsed label or None.
    """
    x0, y0, x1, y1 = structure.plot_bbox
    x_axis_row = structure.x_axis_pixel
    plot_w = max(1, x1 - x0)
    x_end = max(1, int(x0 + 0.25 * plot_w))
    strip = image_bgr[: max(1, x_axis_row), :x_end]
    if strip.size == 0 or strip.shape[0] < 40:
        return None
    # matplotlib y-axis titles are rotated +90 deg (bottom-to-top);
    # a CLOCKWISE rotation turns them back into left-to-right text
    rot = cv2.rotate(strip, cv2.ROTATE_90_CLOCKWISE)
    rot = cv2.resize(rot, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        boxes = ocr.read_text_boxes(rot)
    except Exception:
        return None
    best: Optional[TextBox] = None
    for b in boxes:
        s = b.text.strip()
        if not s:
            continue
        # numeric-only boxes are y tick labels, not the axis title
        try:
            float(s.replace(",", "").replace("%", ""))
            continue
        except ValueError:
            pass
        if best is None or len(b.text) > len(best.text):
            best = b
    if best is None:
        return None
    return _parse_axis_label(best.text)

