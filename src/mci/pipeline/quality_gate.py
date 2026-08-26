"""Quality gate: A/B/C triage with structured, explainable reject reasons.

Design rules (see REAL_ROBUSTNESS_DESIGN.md):
- A: fully automatic extraction (status="ok"), untouched success path.
- B: explainable failure with a minimal human intervention hint
  (multi-panel, too few ticks, categorical axis, low OCR confidence, ...).
- C: unsupported input — refuse explicitly, never emit garbage data
  (no axes, zero ticks, dark background, ...).

The gate *only enhances the failure path*: when the pipeline succeeds,
``QualityVerdict.ok`` is returned and nothing changes for existing callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..schema import (
    AxisFitError,
    CurveExtractionError,
    ExtractionError,
    StructureDetectionError,
    TickReadingError,
)
from .panel_detect import detect_panels

Rect = Tuple[int, int, int, int]

# ---------------------------------------------------------------------------
# Reject / triage reason codes (stable API — tests depend on these strings)
# ---------------------------------------------------------------------------
CODE_STRUCT_MULTI_PANEL = "STRUCT_MULTI_PANEL"      # B: multi-panel layout
CODE_STRUCT_NO_AXIS = "STRUCT_NO_AXIS"              # C: no axis frame found
CODE_STRUCT_DUAL_Y = "STRUCT_DUAL_Y"                # B: two y axes
CODE_OCR_NO_TICKS = "OCR_NO_TICKS"                  # C: zero ticks readable
CODE_OCR_FEW_TICKS = "OCR_FEW_TICKS"                # B: <3 valid ticks
CODE_OCR_LOW_CONF = "OCR_LOW_CONF"                  # B: low OCR confidence
CODE_OCR_CATEGORICAL_AXIS = "OCR_CATEGORICAL_AXIS"  # B: non-numeric axis
CODE_AXIS_TYPE_AMBIGUOUS = "AXIS_TYPE_AMBIGUOUS"    # B: linear/log unclear
CODE_IMG_DARK_BG = "IMG_DARK_BG"                    # C: dark background
CODE_CURVE_EXTRACT_FAIL = "CURVE_EXTRACT_FAIL"      # B: curve stage failed

_HINTS = {
    CODE_STRUCT_MULTI_PANEL: "检测到多个子图面板：请选择单个面板，或使用面板分割后再提取",
    CODE_STRUCT_NO_AXIS: "未检测到坐标轴框线：该图可能无边框（散点/无框布局），暂不支持",
    CODE_STRUCT_DUAL_Y: "检测到双 y 轴：请指定使用左侧或右侧 y 轴",
    CODE_OCR_NO_TICKS: "未能读取任何刻度值：图像过模糊或刻度样式不支持",
    CODE_OCR_FEW_TICKS: "有效刻度不足 3 个：请人工补标 2-3 个刻度值后重试",
    CODE_OCR_LOW_CONF: "刻度 OCR 置信度低：请人工核对刻度值",
    CODE_OCR_CATEGORICAL_AXIS: "该轴为类别轴（非数值刻度）：曲线数据提取无数值意义",
    CODE_AXIS_TYPE_AMBIGUOUS: "线性/对数判型模糊：请人工确认轴类型",
    CODE_IMG_DARK_BG: "深色背景图：当前管线仅支持浅色背景",
    CODE_CURVE_EXTRACT_FAIL: "曲线提取阶段失败：请检查曲线清晰度",
}

# exception type -> default reason code
_EXC_TO_CODE = {
    StructureDetectionError: CODE_STRUCT_NO_AXIS,
    TickReadingError: CODE_OCR_NO_TICKS,
    AxisFitError: CODE_OCR_FEW_TICKS,
    CurveExtractionError: CODE_CURVE_EXTRACT_FAIL,
}


@dataclass
class QualityVerdict:
    """Structured triage result for a (possibly failed) extraction."""

    quality: str  # "A" | "B" | "C"
    status: str  # "ok" | "partial" | "rejected"
    reject_code: Optional[str] = None
    reject_detail: Optional[str] = None
    panels: List[Rect] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _is_dark_background(image_bgr: np.ndarray, white_frac_threshold: float = 0.30) -> bool:
    gray = cv2_gray(image_bgr)
    if gray is None:
        return False
    # fraction of near-white pixels over the whole image
    white = float(np.mean(gray > 200))
    return white < white_frac_threshold


def cv2_gray(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    import cv2

    if image_bgr is None or image_bgr.size == 0:
        return None
    try:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None


def code_for_exception(exc: Exception) -> str:
    """Map an extraction exception to a stable reason code."""
    for exc_type, code in _EXC_TO_CODE.items():
        if isinstance(exc, exc_type):
            return code
    return CODE_CURVE_EXTRACT_FAIL


def classify_failure(
    image_bgr: np.ndarray,
    error: Optional[Exception] = None,
    n_valid_ticks: Optional[int] = None,
    n_predicted_panels: Optional[int] = None,
) -> QualityVerdict:
    """Classify a pipeline failure into A/B/C with an explainable reason.

    Signals:
      - error: the exception raised by the pipeline (None if pre-check).
      - n_valid_ticks: number of ticks with a parsed value (0..N).
      - n_predicted_panels: panel count from ``detect_panels`` (call here or
        pass in; when None, panels are detected internally).
    """
    panels: List[Rect] = []
    n_panels = n_predicted_panels
    if n_panels is None and image_bgr is not None:
        panels = detect_panels(image_bgr)
        n_panels = len(panels)

    # C: unsupported input — dark background (no axes to find anyway)
    if image_bgr is not None and _is_dark_background(image_bgr):
        return QualityVerdict("C", "rejected", CODE_IMG_DARK_BG,
                              _HINTS[CODE_IMG_DARK_BG], panels)

    # B: multi-panel layout is the most common real-world failure (L1)
    if n_panels and n_panels >= 2:
        detail = _HINTS[CODE_STRUCT_MULTI_PANEL] + f"（检测到 {n_panels} 个面板）"
        return QualityVerdict("B", "partial", CODE_STRUCT_MULTI_PANEL, detail, panels)

    # explicit tick shortage (pre-check without an exception)
    if n_valid_ticks is not None and n_valid_ticks == 0:
        return QualityVerdict("C", "rejected", CODE_OCR_NO_TICKS,
                              _HINTS[CODE_OCR_NO_TICKS], panels)
    if n_valid_ticks is not None and 0 < n_valid_ticks < 3:
        return QualityVerdict("B", "partial", CODE_OCR_FEW_TICKS,
                              _HINTS[CODE_OCR_FEW_TICKS], panels)

    # map the exception type
    if error is not None:
        code = code_for_exception(error)
        tier = "C" if code in (CODE_STRUCT_NO_AXIS, CODE_OCR_NO_TICKS,
                               CODE_IMG_DARK_BG) else "B"
        status = "rejected" if tier == "C" else "partial"
        return QualityVerdict(tier, status, code, _HINTS[code], panels)

    # unknown failure without signals -> conservative C (never garbage)
    return QualityVerdict("C", "rejected", CODE_CURVE_EXTRACT_FAIL,
                          _HINTS[CODE_CURVE_EXTRACT_FAIL], panels)


def classify_success(n_warnings: int = 0, axis_quality: Optional[float] = None) -> QualityVerdict:
    """Verdict for a successful extraction.

    Default is "A"/"ok".  When an axis fit quality is supplied and low
    (< 0.95), the result is demoted to "B" with ``AXIS_TYPE_AMBIGUOUS``:
    data has been produced but the linear/log judgement (or fit) is weak and
    deserves a human glance (status stays "ok" — data exists).
    """
    if axis_quality is not None and axis_quality < 0.95:
        return QualityVerdict("B", "ok", CODE_AXIS_TYPE_AMBIGUOUS,
                              _HINTS[CODE_AXIS_TYPE_AMBIGUOUS], [])
    return QualityVerdict("A", "ok", None, None, [])


def is_categorical_axis(ticks) -> bool:
    """True when most tick texts of an axis are non-numeric.

    Used to turn a generic AxisFitError into the explainable
    ``OCR_CATEGORICAL_AXIS`` reason (T2b): dates/months/names are common on
    real charts and must not be silently treated as OCR failures.
    """
    from ..utils import parse_number_text

    texts = [t.text.strip() for t in ticks if t.text and t.text.strip()]
    if len(texts) < 2:
        return False
    non_numeric = sum(1 for t in texts if parse_number_text(t) is None)
    return non_numeric / len(texts) >= 0.7
