"""Phase B-2 tests: axis-title / title / unit recognition."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mci.pipeline.base import TextBox
from mci.pipeline.title_reader import _parse_axis_label, read_titles
from mci.schema import ChartStructure


def _textbox(x0, y0, x1, y1, text):
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    return TextBox(box=box, text=text, score=1.0)


def _structure():
    return ChartStructure(
        plot_bbox=(82, 18, 943, 526),
        x_axis_pixel=527,
        y_axis_pixel=80,
        x_ticks_px=[],
        y_ticks_px=[],
    )


def test_parse_axis_label_units():
    p = _parse_axis_label("Creep strain (%)")
    assert p["variable"] == "Creep strain" and p["unit"] == "%" and p["log_hint"] == ""
    p = _parse_axis_label("Time (h)")
    assert p["variable"] == "Time" and p["unit"] == "h"
    p = _parse_axis_label("Stress / MPa")
    assert p["variable"] == "Stress" and p["unit"] == "MPa"
    p = _parse_axis_label("log Time (s)")
    assert p["log_hint"] == "log"
    p = _parse_axis_label("CreepCurves-BlackWhitePaper")
    assert p["variable"] == "CreepCurves-BlackWhitePaper" and p["unit"] == ""


def test_read_titles_roles():
    struct = _structure()
    boxes = [
        _textbox(340, 17, 671, 37, "Creep Curves - Black White Paper"),  # title
        _textbox(449, 562, 574, 583, "Creep time (h)"),                  # x label
        _textbox(25, 210, 51, 359, "Creep strain (%)"),                  # y label (rotated)
        _textbox(56, 115, 88, 133, "0.8"),                               # y tick label
        _textbox(81, 536, 115, 559, "100"),                              # x tick label
    ]
    out = read_titles(boxes, struct)
    assert out["title"]["text"] == "Creep Curves - Black White Paper"
    assert out["x_label"]["variable"] == "Creep time"
    assert out["x_label"]["unit"] == "h"
    assert out["y_label"]["variable"] == "Creep strain"
    assert out["y_label"]["unit"] == "%"


def test_read_titles_ignores_legend_and_ticks():
    struct = _structure()
    boxes = [
        _textbox(110, 56, 190, 76, "Curve 1"),       # legend inside plot
        _textbox(56, 115, 88, 133, "0.8"),           # y tick label
        _textbox(81, 536, 115, 559, "100"),          # x tick label
    ]
    out = read_titles(boxes, struct)
    assert out == {}


def test_log_prior_from_y_label():
    struct = _structure()
    boxes = [_textbox(25, 210, 51, 359, "log Strain (%)")]
    out = read_titles(boxes, struct)
    assert out["y_label"]["log_hint"] == "log"
