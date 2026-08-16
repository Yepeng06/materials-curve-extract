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


def test_ransac_rejects_misread_tick():
    # 5 ticks of a linear axis; one value misread by OCR (5 -> 30).
    ticks = _ticks_x([0, 25, 50, 75, 100], [0, 2.5, 30, 7.5, 10])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LINEAR
    assert ax.pixel_to_value(0) == pytest.approx(0, abs=0.5)
    assert ax.pixel_to_value(100) == pytest.approx(10, abs=0.5)
    assert ax.quality > 0.99


def test_ransac_log_keeps_geometric_ticks():
    # Log-spaced ticks: the log-space RANSAC must win over the linear one
    # (which would reject everything), keeping the geometric progression.
    ticks = _ticks_x([0, 50, 100, 150], [0.1, 1.0, 10.0, 100.0])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG
    assert ax.pixel_to_value(100) == pytest.approx(10, rel=1e-6)


def test_endpoint_zero_anchor_stabilizes_three_ticks():
    # 3 ticks with one slightly misread; the 0-start anchor pulls the fit
    # back to the true 0..10 mapping over px 0..100.
    ticks = _ticks_x([25, 50, 75], [2.5, 5.0, 7.6])
    ax = fit_axis(ticks, AxisRole.X, endpoint_pixels=(0.0, 100.0))
    assert ax.kind is AxisKind.LINEAR
    assert ax.pixel_to_value(0) == pytest.approx(0, abs=0.15)
    assert ax.pixel_to_value(100) == pytest.approx(10, abs=0.15)


def test_endpoint_anchor_not_applied_when_axis_not_at_zero():
    # Axis spans 20..80: extrapolation to the low endpoint is far from 0,
    # so no anchor may be added.
    ticks = _ticks_x([0, 50, 100], [20, 50, 80])
    ax = fit_axis(ticks, AxisRole.X, endpoint_pixels=(0.0, 100.0))
    assert ax.kind is AxisKind.LINEAR
    assert ax.pixel_to_value(0) == pytest.approx(20, rel=1e-9)
    assert ax.pixel_to_value(100) == pytest.approx(80, rel=1e-9)


def test_endpoint_anchor_not_applied_to_log():
    ticks = _ticks_x([0, 50, 100], [0.1, 1.0, 10.0])
    ax = fit_axis(ticks, AxisRole.X, endpoint_pixels=(0.0, 100.0))
    assert ax.kind is AxisKind.LOG
    assert ax.pixel_to_value(0) == pytest.approx(0.1, rel=1e-6)


def test_endpoint_anchor_needs_positive_span():
    # duplicate values rejected before any anchoring happens
    with pytest.raises(AxisFitError):
        fit_axis(_ticks_x([0, 50, 100], [5, 5, 5]), AxisRole.X,
                 endpoint_pixels=(0.0, 100.0))


# ---------------------------------------------------------------------------
# Phase B-3: multi-signal kind judgement
# ---------------------------------------------------------------------------
def _ticks_x_text(px, texts):
    from mci.schema import Tick
    from mci.utils import parse_number_text

    return [Tick(pixel=p, value=parse_number_text(t), text=t)
            for p, t in zip(px, texts)]


def test_10n_superscript_reread_log_axis():
    # matplotlib log labels "10^-1 .. 10^3" glued by OCR into
    # "0.1", "100", "101", "102", "103" -- the value sequence must be
    # re-resolved to the geometric progression and judged log.
    ticks = _ticks_x_text([80, 286, 492, 698, 904],
                          ["0.1", "100", "101", "102", "103"])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG, ax
    assert ax.pixel_to_value(80) == pytest.approx(0.1, rel=1e-4)
    assert ax.pixel_to_value(904) == pytest.approx(1000, rel=1e-4)


def test_10n_not_reread_on_genuine_linear_axis():
    # "100" on a linear axis is a real value; re-reading it as 10^0=1
    # would break the arithmetic progression and must be rejected.
    ticks = _ticks_x_text([0, 50, 100], ["0", "100", "200"])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LINEAR
    assert ax.pixel_to_value(0) == pytest.approx(0, rel=1e-9)
    assert ax.pixel_to_value(100) == pytest.approx(200, rel=1e-9)


def test_pixel_minor_ticks_vote_log():
    # minor ticks every 5 px, majors at 0/50/100 (spacing ratio 10):
    # dense minors vote log independently of the OCR values
    px = list(range(0, 101, 5))
    ticks = [Tick(pixel=float(p), value=None) for p in px]
    ticks[0].value = 0.1
    ticks[10].value = 1.0
    ticks[20].value = 10.0
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG, ax


def test_pixel_even_majors_only_abstains():
    # majors-only evenly spaced ticks: no spacing evidence -> the R^2
    # fallback decides (linear wins for the 1,2,3 values).
    ticks = _ticks_x_text([0, 50, 100], ["1", "2", "3"])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LINEAR


def test_pixel_uniform_dense_ticks_abstain():
    # uniform dense spacing (ratio 1): undecidable by pixels alone; the
    # value sequence (geometric) must decide log.
    px = list(range(0, 101, 10))
    ticks = [Tick(pixel=float(p), value=None) for p in px]
    ticks[0].value = 0.1
    ticks[5].value = 1.0
    ticks[10].value = 10.0
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG, ax  # value-sequence vote wins


def test_judge_abstains_on_two_ticks():
    from mci.pipeline.axis_kind import judge_axis_kind

    # power-of-ten pair with a power-of-ten ratio -> log (B-3 rule)
    ticks = _ticks_x_text([0, 100], ["0.1", "10"])
    kind, _ = judge_axis_kind(ticks)
    assert kind is AxisKind.LOG
    # non power-of-ten pair -> abstain (R^2 fallback decides)
    ticks = _ticks_x_text([0, 100], ["1.3", "7.9"])
    kind, _ = judge_axis_kind(ticks)
    assert kind is None


def test_judge_hint_prior_wins():
    from mci.pipeline.axis_kind import judge_axis_kind

    ticks = _ticks_x_text([0, 50, 100], ["1", "2", "3"])
    kind, _ = judge_axis_kind(ticks, kind_hint="log")
    assert kind is AxisKind.LOG  # external prior outweighs the value vote
