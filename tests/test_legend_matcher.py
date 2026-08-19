"""Legend matcher tests (multi-curve annotation stage)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np

from mci.pipeline.base import TextBox
from mci.pipeline.legend_matcher import _legend_text_boxes, match_legends
from mci.schema import ChartStructure, Curve


def _structure() -> ChartStructure:
    return ChartStructure(
        plot_bbox=(50, 50, 450, 350),
        x_axis_pixel=350, y_axis_pixel=50,
        x_ticks_px=[100, 200, 300], y_ticks_px=[100, 200, 300],
    )


def _text(text, cx, cy, w=40, h=12):
    return TextBox(np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                             [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]],
                            dtype=np.float64), text, 0.9, anchored=False)


def test_legend_text_boxes_filters_ticks_and_titles():
    s = _structure()
    boxes = [
        _text("0.5", 60, 380),        # tick label (numeric, below axis)
        _text("Creep strain (%)", 60, 200),  # axis title (left)
        _text("700C", 420, 60),       # legend entry (in plot)
        _text("800C", 430, 90),       # legend entry
    ]
    got = _legend_text_boxes(boxes, s)
    texts = [b.text for b in got]
    assert "700C" in texts and "800C" in texts
    assert "0.5" not in texts


def test_match_legends_labels_curves_by_colour():
    s = _structure()
    curves = [
        Curve(name="c0", points=[(0, 0)], pixel_points=[(0, 0)], color=(0, 0, 200)),   # red
        Curve(name="c1", points=[(1, 1)], pixel_points=[(1, 1)], color=(200, 0, 0)),   # blue
    ]
    img = np.full((400, 500, 3), 255, dtype=np.uint8)
    img[54:66, 370:392] = (0, 0, 200)   # red swatch before "Red line"
    img[84:96, 370:392] = (200, 0, 0)   # blue swatch before "Blue line"
    boxes = [
        _text("Red line", 410, 60, w=60),
        _text("Blue line", 410, 90, w=60),
    ]
    out = match_legends(curves, s, boxes, image_bgr=img)
    labels = {c.name: c.legend_label for c in out}
    assert labels["c0"] == "Red line", labels
    assert labels["c1"] == "Blue line", labels


def test_match_legends_noop_without_legend():
    s = _structure()
    curves = [Curve(name="c0", points=[(0, 0)], pixel_points=[(0, 0)], color=(0, 0, 0))]
    assert match_legends(curves, s, [], image_bgr=None) is curves


def test_match_legends_annotates_only_no_drop():
    """Curves are never dropped even when legend colours are unknown."""
    s = _structure()
    curves = [
        Curve(name="c0", points=[(0, 0)], pixel_points=[(0, 0)], color=(0, 0, 200)),
        Curve(name="c1", points=[(1, 1)], pixel_points=[(1, 1)], color=(200, 0, 0)),
    ]
    img = np.full((400, 500, 3), 255, dtype=np.uint8)
    boxes = [_text("Some label", 410, 60, w=60)]
    out = match_legends(curves, s, boxes, image_bgr=img)
    assert len(out) == 2