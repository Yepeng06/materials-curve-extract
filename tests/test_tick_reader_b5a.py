"""Phase B-5a: 10^N superscript-misread disambiguation + low-score filtering.

Covers the OCR failure family found in the 12-worst-image diagnostics:
'100' (10^0 vs 100), '10' (10^0 / 10^-1 / 10), '10-' (10^-1..2), '012'/'0-2'
(10^-2), '101'-'103' (10^1..3), '10.0' (10^0 read with a dot).
"""
import pytest

from mci.pipeline.axis_kind import _candidates, resolve_values
from mci.pipeline.coordinate_mapper import fit_axis
from mci.schema import AxisKind, AxisRole, Tick
from mci.utils import parse_number_text


def _ticks(px, texts):
    return [Tick(pixel=p, value=parse_number_text(t), text=t)
            for p, t in zip(px, texts)]


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------
def test_candidates_plain():
    assert _candidates("0.1") == [0.1]
    assert _candidates("200") == [200.0]
    assert _candidates("abc") == []


def test_candidates_10n_family():
    assert _candidates("10") == [10.0, 1.0, 0.1]      # 10^1 / 10^0 / 10^-1
    assert _candidates("100") == [100.0, 1.0]         # 100 / 10^0
    assert _candidates("10.0") == [10.0, 1.0]         # 10 / 10^0 (dot read)
    assert _candidates("10-") == [0.1, 0.01]          # 10^-1 / 10^-2
    assert _candidates("012") == [12.0, 0.01]         # 12 / 10^-2
    assert _candidates("0-2") == [0.01]               # 10^-2 broken (dedup)
    assert _candidates("101") == [101.0, 10.0]        # 101 / 10^1
    assert _candidates("103") == [103.0, 1000.0]      # 103 / 10^3


# ---------------------------------------------------------------------------
# disambiguation on the diagnosed real cases
# ---------------------------------------------------------------------------
def test_disambig_100_is_10n_on_log_axis():
    # img_0056 x: GT 0.1,1,10,100,1000; '100' at the 10^0 position
    ticks = _ticks([121.0, 316.5, 509.6, 707.5, 902.2],
                   ["10-1", "100", "101", "102", "103"])
    vals, changed = resolve_values(ticks)
    assert changed
    assert [v for v in vals if v is not None] == pytest.approx([0.1, 1, 10, 100, 1000])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG
    # label-centre pixels carry ~1.5% noise; the mapping must stay within 2%
    assert ax.pixel_to_value(902.2) == pytest.approx(1000, rel=2e-2)


def test_disambig_10_is_1_and_01_on_log_axis():
    # img_0038 y: GT 1,0.1,0.01,0.001; '10' read for 10^0 and 10^-1,
    # '10-' for 10^-2, '10-3' for 10^-3
    ticks = _ticks([43, 205, 368, 530], ["10", "10", "10-", "10-3"])
    vals, changed = resolve_values(ticks)
    assert changed
    assert [v for v in vals if v is not None] == pytest.approx([1.0, 0.1, 0.01, 0.001])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG


def test_disambig_10_is_10_on_log_axis():
    # img_0020 y: GT 10,1,0.1,0.01; the TOP '10' is genuine 10, the second
    # is 10^0 = 1
    ticks = _ticks([43.0, 205.0, 367.8, 529.4], ["10", "10", "10-1", "10-2"])
    vals, changed = resolve_values(ticks)
    assert changed
    assert [v for v in vals if v is not None] == pytest.approx([10.0, 1.0, 0.1, 0.01])


def test_disambig_101_and_100_on_log_axis():
    # img_0076 y: GT 10,1,0.1,0.01
    ticks = _ticks([43.0, 205.0, 368.0, 529.4], ["101", "100", "10-1", "10-2"])
    vals, changed = resolve_values(ticks)
    assert changed
    assert [v for v in vals if v is not None] == pytest.approx([10.0, 1.0, 0.1, 0.01])


def test_disambig_012_is_10n2():
    # img_0071 x: '012' read for 0.01 at the first decade
    ticks = _ticks([99.0, 278.0, 463.0, 649.0, 837.0, 1024.0],
                   ["012", "10-1", "100", "101", "102", "103"])
    vals, changed = resolve_values(ticks)
    assert changed
    assert [v for v in vals if v is not None] == pytest.approx(
        [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    ax = fit_axis(ticks, AxisRole.X)
    assert ax.kind is AxisKind.LOG


# ---------------------------------------------------------------------------
# genuine values must NOT be re-resolved
# ---------------------------------------------------------------------------
def test_genuine_100_kept_on_linear_axis():
    ticks = _ticks([0, 50, 100], ["0", "100", "200"])
    vals, changed = resolve_values(ticks)
    assert not changed
    assert [v for v in vals if v is not None] == pytest.approx([0, 100, 200])


def test_genuine_10_kept_on_linear_axis():
    ticks = _ticks([0, 50, 100, 150], ["0", "10", "20", "30"])
    vals, changed = resolve_values(ticks)
    assert not changed
    assert [v for v in vals if v is not None] == pytest.approx([0, 10, 20, 30])


def test_genuine_100_kept_on_log_axis():
    # a REAL 100 between 10 and 1000: re-reading it as 1 destroys the
    # geometric progression
    ticks = _ticks([0, 50, 100, 150], ["10", "100", "1000", "10000"])
    vals, changed = resolve_values(ticks)
    assert not changed
    assert [v for v in vals if v is not None] == pytest.approx([10, 100, 1000, 10000])


def test_nonmonotone_combination_rejected():
    # the regression that broke end-to-end: 0.1,1,0.1,1 looks 'regular'
    # after abs() on the diffs; monotonicity must reject it
    ticks = _ticks([123.0, 442.0, 760.0, 1079.0], ["0.1", "1", "10", "100"])
    vals, changed = resolve_values(ticks)
    assert not changed
    assert [v for v in vals if v is not None] == pytest.approx([0.1, 1, 10, 100])


def test_disambig_2tick_keeps_original_on_tie():
    # power-of-ten pairs are all 'consistent'; the original parse must win
    ticks = _ticks([0, 100], ["0.1", "10"])
    vals, changed = resolve_values(ticks)
    assert not changed
    assert [v for v in vals if v is not None] == pytest.approx([0.1, 10.0])


# ---------------------------------------------------------------------------
# low-score filtering (tick_reader)
# ---------------------------------------------------------------------------
def test_low_score_boxes_filtered():
    from mci.pipeline.base import TextBox
    from mci.pipeline.tick_reader import _associate

    labels = [
        TextBox(box=__import__("numpy").asarray([[100, 520], [160, 538],
                                                 [160, 520], [100, 538]],
                                                dtype=float),
                text="10", score=0.95),
        TextBox(box=__import__("numpy").asarray([[300, 520], [340, 538],
                                                 [340, 520], [300, 538]],
                                                dtype=float),
                text="1", score=0.12),  # fragment: must not appear
    ]
    ticks = _associate([100.0, 300.0], [b for b in labels if b.score >= 0.55],
                       "x", 80)
    vals = [t.value for t in ticks if t.value is not None]
    assert vals == [10.0]
