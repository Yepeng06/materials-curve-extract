"""Tests for export-layer point resampling (web point-count option)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mci.export.resample import resample_points  # noqa: E402


def test_native_passthrough_when_n_zero():
    pts = [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
    assert resample_points(pts, 0) == sorted(pts)


def test_resample_uniform_linear():
    pts = [(0.0, 0.0), (10.0, 10.0)]
    out = resample_points(pts, 5)
    assert len(out) == 5
    xs = [p[0] for p in out]
    assert xs == pytest.approx([0, 2.5, 5, 7.5, 10])
    assert [p[1] for p in out] == pytest.approx(xs)  # y = x line


def test_resample_input_order_irrelevant_and_x_ascending():
    pts = [(3.0, 9.0), (1.0, 1.0), (2.0, 4.0)]
    out = resample_points(pts, 3)
    assert [p[0] for p in out] == pytest.approx([1.0, 2.0, 3.0])
    assert [p[1] for p in out] == pytest.approx([1.0, 4.0, 9.0])


def test_resample_log_axis_uniform_in_decades():
    pts = [(1.0, 1.0), (100.0, 2.0)]
    out = resample_points(pts, 3, x_log=True)
    assert len(out) == 3
    # 3 points over [1, 100] log-space: 1, 10, 100
    assert out[1][0] == pytest.approx(10.0, rel=1e-6)
    assert out[2][0] == pytest.approx(100.0)


def test_resample_drops_nonfinite():
    pts = [(0.0, 0.0), (1.0, float("nan")), (2.0, 4.0)]
    out = resample_points(pts, 4)
    assert len(out) == 4
    # finite part lies on y = 2x
    assert all(p[1] == pytest.approx(2 * p[0]) for p in out)


def test_resample_n_gt_1_required():
    assert resample_points([(1.0, 1.0)], 10) == [(1.0, 1.0)]
    assert resample_points([], 10) == []
