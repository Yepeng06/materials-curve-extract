"""Unified data schema for the chart-curve extraction pipeline.

Every module communicates through these dataclasses only.  Keeping the schema
stable is what lets us swap the classical-CV baseline modules for learned
models (YOLOv8 / U-Net / PaddleOCR) later without touching the rest of the
pipeline — including the future multi-curve extension: ``Curve`` is already a
list inside ``ExtractionResult``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ExtractionError(RuntimeError):
    """Base error for the extraction pipeline."""


class StructureDetectionError(ExtractionError):
    """Chart frame / axis lines could not be detected."""


class TickReadingError(ExtractionError):
    """Tick labels could not be read or associated."""


class AxisFitError(ExtractionError):
    """Not enough valid ticks to fit an axis mapping."""


class CurveExtractionError(ExtractionError):
    """No valid curve component could be extracted."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class AxisKind(str, Enum):
    LINEAR = "linear"
    LOG = "log"


class AxisRole(str, Enum):
    X = "x"
    Y = "y"


# ---------------------------------------------------------------------------
# Axis / tick model
# ---------------------------------------------------------------------------
@dataclass
class Tick:
    """One tick on an axis.

    ``pixel`` is the position along the axis line in image coordinates
    (x for the x-axis, y for the y-axis).  ``value`` is the parsed data value
    (None if OCR failed / not associated yet).
    """

    pixel: float
    value: Optional[float] = None
    text: str = ""
    score: float = 1.0


@dataclass
class AxisSpec:
    """Fitted mapping pixel <-> data value for one axis.

    Mapping model (same for linear and log):
        fitted_space = slope * (sign * pixel) + intercept
        - LINEAR: fitted_space == value
        - LOG:    fitted_space == log10(value)
    ``sign`` is +1 for the x-axis (image x grows with value) and -1 for the
    y-axis (image y grows downward while values grow upward).
    """

    role: AxisRole
    kind: AxisKind
    slope: float
    intercept: float
    vmin: float
    vmax: float
    pmin: float
    pmax: float
    sign: int = 1
    ticks: List[Tick] = field(default_factory=list)
    quality: float = 1.0  # R^2 of the chosen fit, 0..1

    def pixel_to_value(self, pixel: float) -> float:
        v = self.slope * (self.sign * pixel) + self.intercept
        return 10.0 ** v if self.kind is AxisKind.LOG else v

    def value_to_pixel(self, value: float) -> float:
        v = np.log10(value) if self.kind is AxisKind.LOG else value
        return (v - self.intercept) / self.slope / self.sign

    @property
    def span(self) -> float:
        return abs(self.vmax - self.vmin)


# ---------------------------------------------------------------------------
# Chart structure
# ---------------------------------------------------------------------------
@dataclass
class ChartStructure:
    """Rough chart geometry (classical-CV detector output).

    ``plot_bbox`` = (x0, y0, x1, y1) in image pixels; it must fully contain
    the curve pixels.  Precision of the bbox does *not* affect data mapping
    (mapping only uses tick positions), but a too-small bbox clips the curve.
    Later this module is replaced by the YOLOv8 detector, which returns the
    same dataclass.
    """

    plot_bbox: Tuple[int, int, int, int]
    x_axis_pixel: int  # image row of the bottom (x) axis line
    y_axis_pixel: int  # image column of the left (y) axis line
    x_ticks_px: List[float] = field(default_factory=list)  # sorted, ascending
    y_ticks_px: List[float] = field(default_factory=list)  # sorted, ascending
    meta: dict = field(default_factory=dict)  # detector-specific extras

    @property
    def width(self) -> int:
        return self.plot_bbox[2] - self.plot_bbox[0] + 1

    @property
    def height(self) -> int:
        return self.plot_bbox[3] - self.plot_bbox[1] + 1


# ---------------------------------------------------------------------------
# Curve model
# ---------------------------------------------------------------------------
@dataclass
class Curve:
    """One extracted curve.

    Already a standalone object with its own color/label so that the future
    multi-curve backend (multi-component selection + legend matching) only has
    to return more of them.
    """

    name: str
    points: List[Tuple[float, float]]  # data coordinates (x, y)
    pixel_points: List[Tuple[int, int]]  # image coordinates of the trace
    color: Tuple[int, int, int]  # RGB dominant color of the curve pixels
    legend_label: Optional[str] = None


# ---------------------------------------------------------------------------
# End-to-end result
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    image_path: str
    x_axis: Optional[AxisSpec]
    y_axis: Optional[AxisSpec]
    curves: List[Curve]
    structure: Optional[ChartStructure] = None
    meta: dict = field(default_factory=dict)  # timings, per-stage diagnostics
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return (
            self.x_axis is not None
            and self.y_axis is not None
            and len(self.curves) > 0
        )
