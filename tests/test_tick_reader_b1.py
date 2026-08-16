"""Phase B-1 regression tests.

Covers the four tick-reading hardening changes:
  * axis detection robustness (edge-excluded central scan + corner
    connectivity check, bottom-border exclusion),
  * multi-scale (2/5/8 px) tick-band voting,
  * loose region-based label classification (independent of the absolute
    y_axis_col boundary),
  * whole-image OCR fallback when strip OCR yields nothing usable.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mci.pipeline.base import TextBox
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.tick_reader import _classify_labels, read_ticks
from mci.schema import ChartStructure, StructureDetectionError


def _textbox(x0, y0, x1, y1, text, score=1.0):
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return TextBox(box=box, text=text, score=score)


def _structure():
    return ChartStructure(
        plot_bbox=(80, 100, 880, 499),
        x_axis_pixel=500,
        y_axis_pixel=80,
        x_ticks_px=[100.0, 300.0, 500.0, 700.0],
        y_ticks_px=[120.0, 220.0, 320.0, 420.0],
    )


# ---------------------------------------------------------------------------
# label classification (loose region rules)
# ---------------------------------------------------------------------------
def test_classify_accepts_y_labels_when_axis_column_misdetected():
    # y labels sit at cx=40..70 while y_axis_pixel is misdetected at 0:
    # the old absolute rule (cx < y_axis_col - 2) rejected every label.
    struct = _structure()
    struct.plot_bbox = (2, 100, 880, 499)
    struct.y_axis_pixel = 0
    boxes = [
        _textbox(40, 120, 70, 138, "60"),
        _textbox(40, 320, 70, 338, "20"),
        _textbox(100, 520, 160, 538, "10"),   # x label below the axis
        _textbox(350, 30, 600, 50, "Material tensile test"),  # title
    ]
    xl, yl = _classify_labels(boxes, struct)
    assert [b.text for b in xl] == ["10"]
    assert {b.text for b in yl} == {"60", "20"}


def test_classify_title_not_in_x_or_y():
    # A wide centered title must not become an x label (it is above the
    # axis line) nor a y label (it is right of the left 25% region).
    struct = _structure()
    boxes = [
        _textbox(350, 30, 600, 50, "Title"),
        _textbox(100, 520, 160, 538, "10"),
        _textbox(40, 120, 70, 138, "60"),
    ]
    xl, yl = _classify_labels(boxes, struct)
    assert [b.text for b in xl] == ["10"]
    assert [b.text for b in yl] == ["60"]


def test_classify_non_numeric_y_axis_title_is_harmless():
    # An axis title in the left region sneaks into y labels but parses to
    # None downstream, so it only adds a valueless tick (by design).
    struct = _structure()
    boxes = [
        _textbox(10, 200, 34, 400, "Y-axis title"),
        _textbox(40, 120, 70, 138, "60"),
    ]
    xl, yl = _classify_labels(boxes, struct)
    assert [b.text for b in yl] == ["Y-axis title", "60"]


# ---------------------------------------------------------------------------
# whole-image OCR fallback
# ---------------------------------------------------------------------------
def _strip2x(boxes, ox, oy):
    """Convert full-image boxes to 2x upscaled strip-local coordinates,
    which is what the real PaddleOCR backend returns for strip crops
    (_ocr_strips divides by the scale and adds the strip offset)."""

    def conv(b):
        return _textbox((b.box[0][0] - ox) * 2, (b.box[0][1] - oy) * 2,
                        (b.box[2][0] - ox) * 2, (b.box[2][1] - oy) * 2,
                        b.text, b.score)

    return [conv(b) for b in boxes]


class _FakeStripOCR:
    """Mimics PaddleOCRBackend: call 1 = x strip, call 2 = y strip, later
    calls (full image) = whole-image fallback."""

    strip_crops = True

    def __init__(self, x_strip, y_strip, full, x_off=(70, 494), y_off=(0, 80)):
        self.x_strip = _strip2x(x_strip, *x_off)
        self.y_strip = _strip2x(y_strip, *y_off)
        self.full = full
        self.calls = 0

    def read_text_boxes(self, image_bgr):
        self.calls += 1
        if self.calls == 1:
            return self.x_strip
        if self.calls == 2:
            return self.y_strip
        return self.full


def test_whole_image_fallback_when_strips_empty():
    img = np.zeros((600, 960, 3), np.uint8)
    struct = _structure()
    full = [
        _textbox(100, 520, 160, 538, "10"),
        _textbox(40, 120, 70, 138, "60"),
        _textbox(40, 320, 70, 338, "20"),
        _textbox(350, 30, 600, 50, "Title"),
    ]
    ocr = _FakeStripOCR(x_strip=[], y_strip=[], full=full)
    x_ticks, y_ticks = read_ticks(img, struct, ocr, {"tick_assoc_tol_px": 80})
    assert ocr.calls == 3  # x strip + y strip + full-image fallback
    nv_x = sum(1 for t in x_ticks if t.value is not None)
    nv_y = sum(1 for t in y_ticks if t.value is not None)
    assert nv_x == 1
    assert nv_y == 2


def test_no_fallback_when_strips_sufficient():
    # 2 tick marks per axis and both labels read: no whole-image fallback.
    img = np.zeros((600, 960, 3), np.uint8)
    struct = _structure()
    struct.x_ticks_px = [100.0, 700.0]
    struct.y_ticks_px = [120.0, 420.0]
    x_strip = [
        _textbox(100, 520, 160, 538, "10"),
        _textbox(640, 520, 700, 538, "30"),
    ]
    y_strip = [
        _textbox(40, 120, 70, 138, "60"),
        _textbox(40, 320, 70, 338, "20"),
    ]
    ocr = _FakeStripOCR(x_strip=x_strip, y_strip=y_strip, full=x_strip + y_strip)
    x_ticks, y_ticks = read_ticks(img, struct, ocr, {"tick_assoc_tol_px": 80})
    assert ocr.calls == 2  # no full-image call
    assert sum(1 for t in x_ticks if t.value is not None) == 2
    assert sum(1 for t in y_ticks if t.value is not None) == 2


def test_fallback_when_few_read_but_many_marks():
    # 4 tick marks but only 2 labels read on y: the enhanced condition must
    # trigger a whole-image pass (missed labels are likely).
    img = np.zeros((600, 960, 3), np.uint8)
    struct = _structure()
    x_strip = [
        _textbox(100, 520, 160, 538, "10"),
        _textbox(300, 520, 360, 538, "30"),
    ]
    y_strip = [_textbox(40, 120, 70, 138, "60")]
    full = x_strip + y_strip + [_textbox(40, 320, 70, 338, "20")]
    ocr = _FakeStripOCR(x_strip=x_strip, y_strip=y_strip, full=full)
    x_ticks, y_ticks = read_ticks(img, struct, ocr, {"tick_assoc_tol_px": 80})
    assert ocr.calls == 3  # x strip + y strip + full-image fallback
    assert sum(1 for t in y_ticks if t.value is not None) == 2


def test_fallback_not_triggered_for_stub_backend():
    # Stub reads the whole image sidecar already; strip_crops is unset, so
    # the fallback loop must not call OCR twice.
    img = np.zeros((600, 960, 3), np.uint8)
    struct = _structure()

    class _Stub:
        def read_text_boxes(self, image_bgr):
            return [_textbox(100, 520, 160, 538, "10"),
                    _textbox(40, 120, 70, 138, "60")]

    ocr = _Stub()
    x_ticks, y_ticks = read_ticks(img, struct, ocr, {"tick_assoc_tol_px": 80})
    assert sum(1 for t in x_ticks if t.value is not None) == 1


# ---------------------------------------------------------------------------
# axis detection robustness
# ---------------------------------------------------------------------------
def _draw_axes(h, w, axis_x, axis_y, left_edge_ink=True):
    cv2 = pytest.importorskip("cv2")
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.line(img, (axis_x, 0), (axis_x, axis_y), (0, 0, 0), 3)
    cv2.line(img, (0, axis_y), (w - 1, axis_y), (0, 0, 0), 3)
    if left_edge_ink:
        cv2.rectangle(img, (0, 100), (6, axis_y + 20), (0, 0, 0), -1)
    return img


def test_axis_detection_ignores_left_edge_residue():
    # fail_001 scenario: crop residue on the far-left edge must not be
    # detected as the y axis (edge-excluded central scan + connectivity).
    # The 3px line renders with a +/-2px halo; the axis pixel only feeds the
    # strip geometry, so a small offset is acceptable.
    img = _draw_axes(600, 960, axis_x=80, axis_y=500)
    s = detect_structure(img)
    assert abs(s.y_axis_pixel - 80) <= 3, s.y_axis_pixel
    assert abs(s.x_axis_pixel - 500) <= 3, s.x_axis_pixel


def test_axis_detection_ignores_bottom_border():
    import cv2
    img = _draw_axes(600, 960, axis_x=80, axis_y=500, left_edge_ink=False)
    cv2.rectangle(img, (0, 592), (959, 599), (0, 0, 0), -1)  # bottom border
    s = detect_structure(img)
    assert abs(s.x_axis_pixel - 500) <= 3, s.x_axis_pixel
    assert abs(s.y_axis_pixel - 80) <= 3, s.y_axis_pixel


# ---------------------------------------------------------------------------
# multi-scale tick voting
# ---------------------------------------------------------------------------
def test_tick_detection_multi_scale():
    import cv2
    img = _draw_axes(600, 960, axis_x=80, axis_y=500, left_edge_ink=False)
    for x in (150, 300, 450, 600):
        cv2.rectangle(img, (x - 1, 502), (x + 1, 508), (0, 0, 0), -1)
    for y in (150, 250, 350, 450):
        cv2.rectangle(img, (72, y - 1), (78, y + 1), (0, 0, 0), -1)
    s = detect_structure(img)
    assert len(s.x_ticks_px) == 4, s.x_ticks_px
    assert len(s.y_ticks_px) == 4, s.y_ticks_px
    assert abs(s.x_ticks_px[0] - 150) <= 3
    assert abs(s.y_ticks_px[0] - 150) <= 3


def test_tick_detection_rejects_curve_hugging_axis():
    # A curve pixel column touching the bottom axis must not be a tick.
    import cv2
    img = _draw_axes(600, 960, axis_x=80, axis_y=500, left_edge_ink=False)
    cv2.rectangle(img, (149, 502), (151, 508), (0, 0, 0), -1)  # tick-like
    cv2.line(img, (150, 470), (150, 505), (0, 0, 0), 3)        # curve reaches axis
    s = detect_structure(img)
    assert 150.0 not in [round(x) for x in s.x_ticks_px]


def test_no_axes_raises():
    img = np.full((200, 200, 3), 255, np.uint8)
    with pytest.raises(StructureDetectionError):
        detect_structure(img)
