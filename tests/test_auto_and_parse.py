"""Tests for the auto single/multi-curve backend decision + L4 parsing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mci.pipeline import extractor as ex_mod  # noqa: E402
from mci.utils import parse_number_text  # noqa: E402


class _FakeExtractor(ex_mod.Extractor):
    """Bypass __init__ (no config file IO needed) for auto-decision tests."""

    def __init__(self):
        self.cfg = {"multi_mask_thr": 0.35, "max_points": 2000}
        self._single_segmenter = "SINGLE"
        self._multi_segmenter = "MULTI"
        self._segmenter = None


def _auto(monkeypatch, multi_fn, single_fn, cv_fn):
    """Patch module-level curve extractors and run _extract_auto."""
    ex = _FakeExtractor()
    monkeypatch.setattr(ex_mod, "extract_curves_multi", multi_fn)
    monkeypatch.setattr(ex_mod, "extract_curves", single_fn)
    return ex._extract_auto(None, None, None, None)


def test_auto_multi_when_two_or_more(monkeypatch):
    curves, backend = _auto(
        monkeypatch,
        lambda *a, **k: ["a", "b"],
        lambda *a, **k: pytest.fail("single must not run for >=2 channels"),
        lambda *a, **k: pytest.fail("cv must not run for >=2 channels"),
    )
    assert backend == "multi" and len(curves) == 2


def test_auto_single_when_one_channel(monkeypatch):
    curves, backend = _auto(
        monkeypatch,
        lambda *a, **k: ["only"],
        lambda *a, **k: ["refined"],
        lambda *a, **k: pytest.fail("cv must not run"),
    )
    assert backend == "single" and curves == ["refined"]


def test_auto_keeps_multi_single_when_refine_empty(monkeypatch):
    curves, backend = _auto(
        monkeypatch,
        lambda *a, **k: ["only"],
        lambda *a, **k: [],
        lambda *a, **k: pytest.fail("cv must not run"),
    )
    assert backend == "multi" and curves == ["only"]


def test_auto_falls_back_to_cv(monkeypatch):
    def multi(*a, **k):
        raise RuntimeError("multi model missing")

    def single(image, structure, x_axis, y_axis, cfg=None, segmenter=None):
        if segmenter is not None:  # single-curve U-Net path
            raise RuntimeError("single model missing")
        return ["cv-curve"]  # cv fallback path (segmenter=None)

    curves, backend = _auto(monkeypatch, multi, single, None)
    assert backend == "cv" and curves == ["cv-curve"]


def test_auto_propagates_total_failure(monkeypatch):
    def boom(*a, **k):
        raise ex_mod.CurveExtractionError("no curve anywhere")

    with pytest.raises(ex_mod.CurveExtractionError):
        _auto(monkeypatch, boom, boom, boom)


# ---------------- L4 parse extensions ----------------
def test_parse_unit_suffix():
    assert parse_number_text("10 MPa") == 10.0
    assert parse_number_text("10MPa") == 10.0
    assert parse_number_text("0.5 mm") == 0.5
    assert parse_number_text("25 °C") == 25.0
    assert parse_number_text("2h") == 2.0


def test_parse_prefix_markers():
    assert parse_number_text("~25") == 25.0
    assert parse_number_text(">100") == 100.0
    assert parse_number_text("≤0.5") == 0.5
    assert parse_number_text("±0.5") == 0.5


def test_parse_confusables():
    assert parse_number_text("l0") == 10.0
    assert parse_number_text("1O0") == 100.0
    assert parse_number_text("S5") == 55.0  # all-confusable token rescue


def test_parse_still_rejects_words():
    assert parse_number_text("Creep") is None
    assert parse_number_text("") is None
