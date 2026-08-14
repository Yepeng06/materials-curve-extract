"""Pixel -> data coordinate mapping with automatic linear/log axis detection.

For each axis we have tick positions (pixels) with parsed values.  We fit two
models and pick the better one by residual / R^2:

  * linear:  value = a * p + b
  * log:     log10(value) = a * p + b        (requires positive values)

where p is the position along the axis oriented so that values increase with
p (x-axis: image x; y-axis: negated image y).  This fit-comparison is the
baseline for the "coordinate type judgement" module; in the full project it
can be replaced/augmented by a dedicated classifier, but the residuals give
a hard statistical guarantee and are cheap.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schema import AxisKind, AxisRole, AxisSpec, AxisFitError, Tick


def _fit(p: np.ndarray, v: np.ndarray) -> Tuple[float, float, float, float]:
    """Least-squares line fit; returns (slope, intercept, R^2, RMS error)."""
    a, b = np.polyfit(p, v, 1)
    pred = a * p + b
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    rms = float(np.sqrt(np.mean((v - pred) ** 2))) if len(v) else float("inf")
    return float(a), float(b), r2, rms


def fit_axis(ticks: List[Tick], role: AxisRole, kind_hint: str = "auto",
             min_ticks: int = 2) -> AxisSpec:
    """Fit one axis mapping from valued ticks.

    kind_hint: "auto" | "linear" | "log"  (config-level override / prior).
    """
    valued = [(t.pixel, t.value) for t in ticks if t.value is not None]
    if len(valued) < min_ticks:
        raise AxisFitError(
            f"{role.value}-axis: only {len(valued)} readable ticks "
            f"(need >= {min_ticks})"
        )
    p_raw = np.array([px for px, _ in valued], dtype=np.float64)
    v = np.array([vv for _, vv in valued], dtype=np.float64)
    if len(v) >= 2 and float(np.ptp(v)) < 1e-9:
        raise AxisFitError(f"{role.value}-axis: tick values are all equal (constant axis?)")
    sign = 1 if role is AxisRole.X else -1
    p = sign * p_raw

    a_lin, b_lin, r2_lin, rms_lin = _fit(p, v)

    ok_log = False
    a_log = b_log = r2_log = rms_log = float("inf")
    pos = v > 0
    if pos.sum() >= min_ticks:
        a_log, b_log, r2_log, rms_log = _fit(p[pos], np.log10(v[pos]))
        ok_log = r2_log > 0.9 and np.isfinite(rms_log)

    # decide axis kind
    kind = AxisKind.LINEAR
    if kind_hint == "log" and ok_log:
        kind = AxisKind.LOG
    elif kind_hint == "linear":
        kind = AxisKind.LINEAR
    elif kind_hint == "auto":
        # log wins when it is clearly better: (a) its R^2 beats linear's by a
        # wide margin, or (b) linear is not already near-perfect and its RMS
        # is > 4x log's.  When both fits are essentially perfect (e.g. only
        # 2 ticks, where any model interpolates exactly) linear wins — the
        # common case in papers — and a warning is surfaced upstream.
        if ok_log and (
            r2_log > r2_lin + 0.01
            or (r2_lin < 0.999 and rms_log < 0.25 * rms_lin)
        ):
            kind = AxisKind.LOG

    if kind is AxisKind.LOG:
        slope, intercept, quality = a_log, b_log, r2_log
    else:
        slope, intercept, quality = a_lin, b_lin, r2_lin

    return AxisSpec(
        role=role,
        kind=kind,
        slope=slope,
        intercept=intercept,
        vmin=float(v.min()),
        vmax=float(v.max()),
        pmin=float(p_raw.min()),
        pmax=float(p_raw.max()),
        sign=sign,
        ticks=list(ticks),
        quality=quality,
    )


def build_axes(
    x_ticks: List[Tick],
    y_ticks: List[Tick],
    x_hint: str = "auto",
    y_hint: str = "auto",
) -> Tuple[AxisSpec, AxisSpec]:
    return (
        fit_axis(x_ticks, AxisRole.X, x_hint),
        fit_axis(y_ticks, AxisRole.Y, y_hint),
    )
