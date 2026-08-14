"""Unit tests for axis fitting / coordinate mapping."""
import pytest

from mci.pipeline.coordinate_mapper import fit_axis
from mci.schema import AxisKind, AxisRole, AxisFitError, Tick


def _ticks_x(px, vals):
    return [Tick(pixel=p, value=v) for p, v in zip(px, vals)]


def test_linear_x():
    # pixel 0..100 -> value 0..10
    ticks = _ticks_x([0, 50, 100], [0, 5, 10])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LINEAR
    assert ax.pixel_to_value(0) == pytest.approx(0)
    assert ax.pixel_to_value(100) == pytest.approx(10)
    assert ax.pixel_to_value(25) == pytest.approx(2.5)
    assert ax.quality > 0.999


def test_linear_x_with_offset():
    ticks = _ticks_x([10, 60, 110], [100, 200, 300])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LINEAR
    assert ax.pixel_to_value(10) == pytest.approx(100, rel=1e-9)
    assert ax.pixel_to_value(110) == pytest.approx(300, rel=1e-9)


def test_linear_y_inverted():
    # y axis: image y grows downward; value grows upward -> slope negative
    ticks = _ticks_x([10, 50, 90], [90, 50, 10])
    ax = fit_axis(ticks, AxisRole.Y)
    assert ax.kind is AxisKind.LINEAR
    assert ax.pixel_to_value(10) == pytest.approx(90, rel=1e-9)
    assert ax.pixel_to_value(90) == pytest.approx(10, rel=1e-9)
    assert ax.value_to_pixel(50) == pytest.approx(50, abs=1e-6)


def test_log_x_auto_detected():
    # log-spaced ticks at 0.1, 1, 10, 100 over exactly log-spaced pixels
    ticks = _ticks_x([0, 100 / 3, 200 / 3, 100], [0.1, 1.0, 10.0, 100.0])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG
    assert ax.pixel_to_value(0) == pytest.approx(0.1, rel=1e-9)
    assert ax.pixel_to_value(100) == pytest.approx(100, rel=1e-9)
    assert ax.pixel_to_value(100 / 3) == pytest.approx(1.0, rel=1e-6)
    assert ax.pixel_to_value(50) == pytest.approx(10 ** 0.5, rel=1e-6)


def test_log_y():
    ticks = _ticks_x([100, 50, 0], [100.0, 1.0, 0.01])  # y image px 100->value 100
    ax = fit_axis(ticks, AxisRole.Y)
    assert ax.kind is AxisKind.LOG
    assert ax.pixel_to_value(100) == pytest.approx(100, rel=1e-9)
    assert ax.pixel_to_value(0) == pytest.approx(0.01, rel=1e-9)


def test_hint_log_overrides():
    ticks = _ticks_x([0, 50, 100], [1, 10, 100])  # also fits linear (1,10,100)
    ax = fit_axis(ticks, AxisRole.X, kind_hint="log")
    assert ax.kind is AxisKind.LOG
    assert ax.pixel_to_value(0) == pytest.approx(1, rel=1e-9)


def test_linear_preferred_when_ambiguous():
    # values 1,2,3 fit both models perfectly; linear must win
    ticks = _ticks_x([0, 50, 100], [1, 2, 3])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LINEAR


def test_too_few_ticks():
    with pytest.raises(AxisFitError):
        fit_axis(_ticks_x([0, 100], [None, 5]), AxisRole.X)
    with pytest.raises(AxisFitError):
        fit_axis(_ticks_x([0, 100], [1, None]), AxisRole.X)


def test_duplicate_values_rejected():
    with pytest.raises(AxisFitError):
        fit_axis(_ticks_x([0, 50, 100], [5, 5, 5]), AxisRole.X)


def test_roundtrip():
    import numpy as np

    ticks = _ticks_x([0, 25, 50, 75, 100], [0, 2.5, 5, 7.5, 10])
    ax = fit_axis(ticks, AxisRole.X)
    for p, v in zip([0, 25, 50, 75, 100], [0, 2.5, 5, 7.5, 10]):
        assert ax.value_to_pixel(v) == pytest.approx(p, abs=1e-6)
        assert ax.pixel_to_value(p) == pytest.approx(v, abs=1e-6)
