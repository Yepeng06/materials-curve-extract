"""Unit tests for multi-panel detection (REAL_ROBUSTNESS_DESIGN.md S1)."""
import numpy as np
import pytest

from mci.pipeline.panel_detect import detect_panels, panel_grid_shape


def _render(rects, w=600, h=400, line_w=3, bg=255, fg=0):
    """Render white image with black rectangles (x0, y0, x1, y1)."""
    img = np.full((h, w), bg, dtype=np.uint8)
    img = np.stack([img] * 3, axis=-1)
    for (x0, y0, x1, y1) in rects:
        img[y0:y1 + 1, x0:x0 + line_w] = fg
        img[y0:y1 + 1, x1 - line_w + 1:x1 + 1] = fg
        img[y0:y0 + line_w, x0:x1 + 1] = fg
        img[y1 - line_w + 1:y1 + 1, x0:x1 + 1] = fg
    return img


def test_single_panel_no_extra():
    img = _render([(60, 50, 540, 350)])
    panels = detect_panels(img)
    assert len(panels) == 1
    x0, y0, x1, y1 = panels[0]
    assert abs(x0 - 60) <= 3 and abs(y0 - 50) <= 3


def test_two_panels_horizontal():
    img = _render([(40, 50, 290, 350), (310, 50, 560, 350)])
    panels = detect_panels(img)
    assert len(panels) == 2
    # deterministic left-to-right order
    assert panels[0][0] < panels[1][0]


def test_two_panels_vertical():
    img = _render([(60, 30, 540, 190), (60, 210, 540, 370)])
    panels = detect_panels(img)
    assert len(panels) == 2


def test_nested_frame_collapsed():
    # outer full-frame + inner plot frame -> only inner kept
    img = _render([(10, 10, 590, 390), (80, 70, 520, 330)])
    panels = detect_panels(img)
    assert len(panels) == 1
    x0, y0, _, _ = panels[0]
    assert abs(x0 - 80) <= 3


def test_grid_lines_ignored():
    # a single panel with many internal grid lines (no closed rects)
    img = _render([(60, 50, 540, 350)])
    for gx in range(120, 520, 60):
        img[:, gx:gx + 1] = 200
    for gy in range(100, 340, 50):
        img[gy:gy + 1, :] = 200
    panels = detect_panels(img)
    assert len(panels) == 1


def test_blank_image_empty():
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    assert detect_panels(img) == []


def test_dark_image_empty():
    img = np.full((300, 500, 3), 20, dtype=np.uint8)
    assert detect_panels(img) == []


def test_too_small_rect_ignored():
    img = _render([(60, 50, 540, 350), (250, 180, 270, 200)])  # tiny inner box
    panels = detect_panels(img)
    assert len(panels) == 1


def test_grid_shape_inference():
    panels = [(40, 50, 290, 190), (310, 50, 560, 190),
              (40, 210, 290, 350), (310, 210, 560, 350)]
    assert panel_grid_shape(panels) == (2, 2)
    assert panel_grid_shape(panels[:3]) is None  # irregular
    assert panel_grid_shape([]) is None
