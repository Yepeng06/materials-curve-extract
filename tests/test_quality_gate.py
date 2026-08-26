"""Unit tests for the A/B/C quality gate (REAL_ROBUSTNESS_DESIGN.md S1)."""
import numpy as np
import pytest

from mci.pipeline.quality_gate import (
    CODE_AXIS_TYPE_AMBIGUOUS,
    CODE_IMG_DARK_BG,
    CODE_OCR_FEW_TICKS,
    CODE_OCR_NO_TICKS,
    CODE_STRUCT_MULTI_PANEL,
    CODE_STRUCT_NO_AXIS,
    classify_failure,
    classify_success,
    code_for_exception,
    is_categorical_axis,
)
from mci.schema import AxisFitError, StructureDetectionError, Tick, TickReadingError


def _white(w=600, h=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _dark(w=600, h=400):
    return np.full((h, w, 3), 15, dtype=np.uint8)


def _with_panels(img, n=2):
    out = img.copy()
    h, w = out.shape[:2]
    ww, hh = w // n, h
    for i in range(n):
        x0, x1 = i * ww + 8, (i + 1) * ww - 8
        out[6:9, x0:x1 + 1] = 0
        out[h - 9:h - 6, x0:x1 + 1] = 0
        out[6:h - 6, x0:x0 + 3] = 0
        out[6:h - 6, x1 - 2:x1 + 1] = 0
    return out


def test_success_is_a():
    v = classify_success()
    assert v.ok and v.quality == "A" and v.status == "ok"
    assert v.reject_code is None


def test_multi_panel_is_b():
    img = _with_panels(_white())
    v = classify_failure(img, error=StructureDetectionError("no axis"))
    assert v.quality == "B" and v.status == "partial"
    assert v.reject_code == CODE_STRUCT_MULTI_PANEL
    assert len(v.panels) == 2
    assert v.reject_detail


def test_dark_background_is_c():
    v = classify_failure(_dark(), error=StructureDetectionError("no axis"))
    assert v.quality == "C" and v.status == "rejected"
    assert v.reject_code == CODE_IMG_DARK_BG


def test_no_ticks_is_c():
    v = classify_failure(_white(), n_valid_ticks=0)
    assert v.quality == "C" and v.reject_code == CODE_OCR_NO_TICKS


def test_few_ticks_is_b():
    v = classify_failure(_white(), n_valid_ticks=2)
    assert v.quality == "B" and v.reject_code == CODE_OCR_FEW_TICKS


def test_exception_mapping():
    assert code_for_exception(StructureDetectionError("x")) == CODE_STRUCT_NO_AXIS
    assert code_for_exception(TickReadingError("x")) == CODE_OCR_NO_TICKS
    assert code_for_exception(AxisFitError("x")) == CODE_OCR_FEW_TICKS


def test_unknown_failure_maps_to_curve_fail_b():
    # non-pipeline exceptions map to CURVE_EXTRACT_FAIL (B: internal error)
    v = classify_failure(_white(), error=RuntimeError("boom"))
    assert v.quality == "B" and v.status == "partial"
    assert v.reject_code == "CURVE_EXTRACT_FAIL"


def test_verdict_defaults():
    v = classify_failure(_white(), n_valid_ticks=None)
    assert v.quality in ("B", "C")


def test_success_low_axis_quality_is_b():
    v = classify_success(axis_quality=0.90)
    assert v.quality == "B" and v.status == "ok"
    assert v.reject_code == CODE_AXIS_TYPE_AMBIGUOUS


def test_success_high_axis_quality_is_a():
    assert classify_success(axis_quality=0.99).quality == "A"


def test_is_categorical_axis():
    def tks(texts):
        return [Tick(pixel=float(i), text=t) for i, t in enumerate(texts)]

    assert is_categorical_axis(tks(["Jan", "Feb", "Mar", "Apr"])) is True
    assert is_categorical_axis(tks(["2018", "2019", "2020", "2021"])) is False
    assert is_categorical_axis(tks(["1.0", "2.0", "3.0"])) is False
    assert is_categorical_axis(tks(["0.1", "0.01", "0.001"])) is False
    assert is_categorical_axis(tks(["Jan", "Feb"])) is True
    assert is_categorical_axis(tks([])) is False
    assert is_categorical_axis(tks(["only"])) is False
