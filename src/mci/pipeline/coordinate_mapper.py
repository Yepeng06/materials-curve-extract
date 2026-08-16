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

Phase B-1 hardening:
  * RANSAC outlier rejection before fitting (misread tick values are the
    dominant real-OCR failure mode): with >= 4 valued ticks we run RANSAC in
    both the linear and the log10 space, keep the space with more inliers
    and drop the outliers (at least 3 ticks are always kept).
  * Endpoint anchor: when the axis plausibly starts at 0 (the extrapolated
    value at the low-value axis endpoint is within 2% of the tick span) the
    endpoint pixel is anchored at value 0 and joins the fit.  This stabilizes
    the mapping in the common 2-tick case, where a single misread destroys
    the fit (any model interpolates 2 points exactly).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schema import AxisKind, AxisRole, AxisSpec, AxisFitError, Tick
from .axis_kind import judge_axis_kind, resolve_values


def _fit(p: np.ndarray, v: np.ndarray) -> Tuple[float, float, float, float]:
    """Least-squares line fit; returns (slope, intercept, R^2, RMS error)."""
    a, b = np.polyfit(p, v, 1)
    pred = a * p + b
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    rms = float(np.sqrt(np.mean((v - pred) ** 2))) if len(v) else float("inf")
    return float(a), float(b), r2, rms


def _ransac_inliers(
    p: np.ndarray, v: np.ndarray, n_iter: int = 100,
    tol_frac: float = 0.05, seed: int = 12345,
) -> np.ndarray:
    """RANSAC inlier mask for the 1-D linear model v = a * p + b.

    A position is an inlier when its residual is within tol_frac of the
    value span.  Returns all-True for fewer than 4 points (no rejection).
    """
    n = len(p)
    inl = np.ones(n, dtype=bool)
    if n < 4:
        return inl
    span = float(np.ptp(v))
    if span <= 1e-12:
        return inl
    tol = tol_frac * span
    rng = np.random.default_rng(seed)
    best: Optional[np.ndarray] = None
    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        if abs(p[i] - p[j]) < 1e-9:
            continue
        a = (v[i] - v[j]) / (p[i] - p[j])
        b = v[i] - a * p[i]
        m = np.abs(a * p + b - v) <= tol
        if best is None or int(m.sum()) > int(best.sum()):
            best = m
            if int(best.sum()) == n:
                break
    return best if best is not None else inl


def _fit_with_endpoint_zero(
    p: np.ndarray, v: np.ndarray, sign: int,
    endpoint_pixels: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Anchor the low-value axis endpoint at 0 when extrapolation supports it.

    Returns (p, v) with the anchor appended and a flag saying whether the
    anchor was added.  The low-value endpoint in image pixels is the min
    pixel for the x axis (sign=+1) and the max pixel for the y axis
    (sign=-1).  The anchor is only added when the provisional linear fit
    extrapolates to within 2% of the tick value span - i.e. the axis
    plausibly starts at 0 (extremely common in data plots).
    """
    if endpoint_pixels is None or len(v) < 2:
        return p, v, False
    span_v = float(np.ptp(v))
    if span_v <= 1e-9:
        return p, v, False
    # When the ticks already include a value at (or very near) 0, the axis
    # start is anchored by that tick itself — adding the endpoint pixel
    # would only inject the label-center vs axis-line pixel offset as a
    # systematic error.
    if abs(float(v.min())) <= 0.02 * span_v:
        return p, v, False
    e0, e1 = sorted(float(x) for x in endpoint_pixels)
    low_px = e0 if sign == 1 else e1
    a0, b0 = np.polyfit(p, v, 1)
    resid = a0 * p + b0 - v
    v_low = a0 * (sign * low_px) + b0
    if len(v) == 2:
        # no residual degrees of freedom: require an absolute agreement
        if abs(v_low) > 0.005 * span_v:
            return p, v, False
    else:
        # statistical test: 0 must lie within ~2 sigma of the extrapolated
        # low-end value (an axis that really starts at 0 extrapolates to
        # ~0 within the fit's own noise; e.g. a 0.1..0.7 axis extrapolates
        # to ~0.006 and must NOT be anchored)
        sigma = float(np.sqrt(np.sum(resid ** 2) / (len(v) - 2)))
        denom = float(np.sum((p - p.mean()) ** 2))
        se_low = sigma * float(np.sqrt(1.0 / len(v) + (sign * low_px - p.mean()) ** 2 / denom)) if denom > 1e-9 else float("inf")
        if se_low <= 1e-12 or abs(v_low) > 2.0 * se_low:
            return p, v, False
    return (
        np.concatenate([p, [sign * low_px]]),
        np.concatenate([v, [0.0]]),
        True,
    )


def fit_axis(ticks: List[Tick], role: AxisRole, kind_hint: str = "auto",
             min_ticks: int = 2,
             endpoint_pixels: Optional[Tuple[float, float]] = None) -> AxisSpec:
    """Fit one axis mapping from valued ticks.

    kind_hint: "auto" | "linear" | "log"  (config-level override / prior).
    endpoint_pixels: (low, high) axis endpoint pixels in image coordinates
    (x: left/right of the axis line; y: top/bottom of the plot).  Used for
    the 0-start endpoint anchor (see above).

    Phase B-3: the axis kind is decided by the multi-signal judge
    (value-sequence consistency incl. 10N superscript re-resolution,
    pixel-spacing minor-tick density, external prior); R^2 only grades the
    chosen mapping and serves as the fallback when the judge abstains.
    """
    values, _reread = resolve_values(ticks)
    valued = [(t.pixel, v, t.score) for t, v in zip(ticks, values) if v is not None]
    # de-duplicate near-coincident ticks (<3 px apart: real OCR can emit two
    # boxes for one label, e.g. '200' and '0' on the same spot -- the extra
    # value pollutes the sequence checks); keep the higher-scoring one
    valued.sort(key=lambda x: x[0])
    deduped = []
    for p, v, s in valued:
        if deduped and abs(p - deduped[-1][0]) < 3.0:
            if s > deduped[-1][2]:
                deduped[-1] = (p, v, s)
        else:
            deduped.append((p, v, s))
    valued = [(p, v) for p, v, _ in deduped]
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

    # ---- single-outlier rejection for 3 ticks ----
    # 3 valued ticks with one misread defeat the sequence vote (any 2 of 3
    # fit perfectly); when the 3-value sequence is inconsistent, drop the
    # tick whose removal leaves a near-perfect progression.
    if len(v) == 3:
        from .axis_kind import _seq_scores

        s3 = _seq_scores(v.tolist())
        if s3 is not None and min(s3) > 0.1:
            for drop in range(3):
                v2_ = np.delete(v, drop)
                p2_ = np.delete(p, drop)
                s2 = _seq_scores(v2_.tolist())
                if s2 is not None and min(s2) < 0.05:
                    p, v = p2_, v2_
                    break

    # ---- multi-signal kind judgement (B-3) ----
    judged, evidence = judge_axis_kind(ticks, kind_hint)

    # ---- RANSAC outlier rejection in the judged space (or both when
    # the judge abstained); at least 3 ticks are always kept ----
    if len(v) >= 4:
        if judged is AxisKind.LOG:
            pos = v > 0
            keep = np.ones(len(v), dtype=bool)
            if int(pos.sum()) >= 4:
                keep[pos] = _ransac_inliers(p[pos], np.log10(v[pos]))
            keep = keep if int(keep.sum()) >= 3 else np.ones(len(v), dtype=bool)
        elif judged is AxisKind.LINEAR:
            keep = _ransac_inliers(p, v)
        else:
            inl_lin = _ransac_inliers(p, v)
            inl_log = np.zeros(len(v), dtype=bool)
            pos = v > 0
            if int(pos.sum()) >= 4:
                inl_log[pos] = _ransac_inliers(p[pos], np.log10(v[pos]))
            keep = inl_lin if int(inl_lin.sum()) >= int(inl_log.sum()) else inl_log
        if int(keep.sum()) >= 3 and int(keep.sum()) < len(v):
            p = p[keep]
            v = v[keep]

    # ---- endpoint anchor: axis starts near 0 (linear, low end) ----
    p, v, anchored = _fit_with_endpoint_zero(p, v, sign, endpoint_pixels)

    a_lin, b_lin, r2_lin, rms_lin = _fit(p, v)

    ok_log = False
    a_log = b_log = r2_log = rms_log = float("inf")
    pos = v > 0
    if int(pos.sum()) >= min_ticks:
        a_log, b_log, r2_log, rms_log = _fit(p[pos], np.log10(v[pos]))
        ok_log = r2_log > 0.9 and np.isfinite(rms_log)

    # decide axis kind: judge's vote wins; R^2 double-fit is the fallback
    kind = judged
    if kind is None:
        kind = AxisKind.LINEAR
        if kind_hint == "log" and ok_log:
            kind = AxisKind.LOG
        elif kind_hint == "linear":
            kind = AxisKind.LINEAR
        elif kind_hint == "auto":
            # log wins when it is clearly better: (a) its R^2 beats linear's
            # by a wide margin, or (b) linear is not already near-perfect and
            # its RMS is > 4x log's.  When both fits are essentially perfect
            # (e.g. only 2 ticks, where any model interpolates exactly)
            # linear wins -- the common case in papers.
            if ok_log and (
                r2_log > r2_lin + 0.01
                or (r2_lin < 0.999 and rms_log < 0.25 * rms_lin)
            ):
                kind = AxisKind.LOG

    if kind is AxisKind.LOG and ok_log:
        slope, intercept, quality = a_log, b_log, r2_log
    else:
        kind = AxisKind.LINEAR
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
    x_endpoints: Optional[Tuple[float, float]] = None,
    y_endpoints: Optional[Tuple[float, float]] = None,
) -> Tuple[AxisSpec, AxisSpec]:
    return (
        fit_axis(x_ticks, AxisRole.X, x_hint, endpoint_pixels=x_endpoints),
        fit_axis(y_ticks, AxisRole.Y, y_hint, endpoint_pixels=y_endpoints),
    )

