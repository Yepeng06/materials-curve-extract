"""Unit tests for evaluation metrics."""
import numpy as np
import pytest

from mci.eval.metrics import curve_metrics, summarize


def test_exact_match_zero_rmse():
    pts = [(float(i), float(2 * i)) for i in range(100)]
    m = curve_metrics(pts, pts)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["rel_rmse"] == pytest.approx(0.0)
    assert m["x_coverage"] == pytest.approx(1.0)
    assert m["n_pred"] == 100


def test_constant_offset():
    gt = [(float(i), float(i)) for i in range(0, 100)]
    pred = [(float(i), float(i) + 0.5) for i in range(0, 100)]
    m = curve_metrics(gt, pred)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["rel_rmse"] == pytest.approx(0.5 / 99.0, rel=1e-9)


def test_interpolation_at_predicted_x():
    # pred samples on a denser / different grid -> GT interpolated
    gt = [(0.0, 0.0), (10.0, 10.0)]
    pred = [(5.0, 5.0), (6.0, 6.0)]
    m = curve_metrics(gt, pred)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_coverage_penalty():
    gt = [(float(i), float(i)) for i in range(0, 100)]
    pred = [(float(i), float(i)) for i in range(0, 50)]
    m = curve_metrics(gt, pred)
    assert m["x_coverage"] == pytest.approx(49 / 99)
    assert m["rmse"] == pytest.approx(0.0)


def test_too_few_points():
    m = curve_metrics([(0.0, 0.0)], [(0.0, 0.1), (1.0, 1.1)])
    assert np.isnan(m["rmse"])


def test_summarize():
    rows = [
        {"status": "ok", "rel_rmse": 0.005},
        {"status": "ok", "rel_rmse": 0.008},
        {"status": "ok", "rel_rmse": 0.02},
        {"status": "failed", "error": "x"},
    ]
    s = summarize(rows)
    assert s["n_images"] == 4
    assert s["n_failed"] == 1
    assert s["rel_rmse_mean"] == pytest.approx(0.011, rel=1e-6)
    assert s["pass_rate_1pct"] == pytest.approx(2 / 3)
