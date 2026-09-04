"""Curve redraw reconstruction: turn noisy extracted points into a clean,
shape-faithful curve.

Problem
-------
The extractor returns per-curve ``points`` (data coordinates) mapped from the
pixel chain.  Tracing/fallback artifacts inject "bad points" (错点): isolated
spikes (a single point far from the curve), jumps (the tracer briefly rides
another curve / grid remnant / drifted tail), and pixel-quantization jitter.
Plotting these raw distort the redrawn curve and hide the true material shape
(creep stages, yield points).

Method (researched, see ``重绘_research.md`` for the argumentation)
-------------------------------------------------------------------
1. **Adaptive Hampel (median + MAD) spike rejection** — for each coordinate a
   point is flagged when it deviates from a local median by more than
   ``k*MAD``.  Unlike mean-based methods this is immune to the spike's own
   magnitude (outlier-robust; MAD ~ robust sigma), so genuine sharp kinks are
   kept while isolated spikes are dropped [Pearson et al. 2016; Hampel 1974].
2. **Sustained-segment-jump rejection** (second stage) — a *flat, sustained*
   offset run (tracer riding a neighbouring curve for many points) is invisible
   to Hampel because the whole local window is shifted.  We fit a local trend
   that follows genuine kinks, then flag runs that are both long (``>=
   jump_min_run``) and flat (within-run residual CV ``< jump_flat_thr``) — a
   *peaked* residual (genuine kink) is excluded, a *flat* residual (jump) is
   removed.  Residual-control-chart / CUSUM run-length logic [Lund Univ. robust
   residual control charts 2020].
3. **Robust LOWESS (bisquare reweighted local linear regression)** with
   robustness iterations to remove pixel-quantization *jitter* while turning
   at genuine kinks — it down-weights high-residual points instead of cutting
   a square smoothing window across a sharp feature [Cleveland 1979;
   statsmodels implementation].  Fallback to a local-median smoother when
   statsmodels is unavailable.
4. Smoothing is done **parametrically against path order ``t``** for *both*
   coordinates, so non-monotonic / near-vertical / serrated segments are not
   distorted (a y=f(t) treatment alone would break on vertical runs).
5. Optional **monotone isotonic projection** for axes where physical
   monotonicity is guaranteed (e.g. creep strain vs time).  **Off by default**
   — must not be applied to serrating stress–strain or non-monotonic
   polarization.

A log axis is transformed to its fitted (log10) space before rejection +
smoothing so that "a few percent" has the same meaning across the range, then
transformed back.

This module deliberately does NOT touch the extracted ``points`` used for the
RMSE acceptance metric; it only cleans the *rendering* of the redraw.
"""
from __future__ import annotations

import numpy as np

try:  # statsmodels -> robust LOWESS (bisquare reweighting)
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    _HAS_LOWESS = True
except Exception:  # noqa: BLE001
    _HAS_LOWESS = False


# ---------------------------------------------------------------------------
# 1. Adaptive Hampel / median + MAD spike rejection
# ---------------------------------------------------------------------------
def _hampel_mask(sig: np.ndarray, window: int = 9, k: float = 4.0) -> np.ndarray:
    """Return a bool mask marking non-outliers.

    For each index i, compare sig[i] with the median of a centred window and
    the window's MAD (scaled to sigma via 1.4826).  A point whose absolute
    deviation exceeds ``k`` times that robust sigma is an outlier.
    ``window`` should be odd and >= 3.
    """
    n = len(sig)
    good = np.ones(n, dtype=bool)
    if n < 5:
        return good
    half = int(window) // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = sig[lo:hi]
        med = np.median(w)
        mad = float(np.median(np.abs(w - med)) * 1.4826)  # robust sigma
        if mad <= 1e-12:
            continue  # window is (near-)constant: nothing to flag
        if abs(sig[i] - med) > k * mad:
            good[i] = False
    return good


# ---------------------------------------------------------------------------
# 2. Robust smoothing (LOWESS, fallback local median)
# ---------------------------------------------------------------------------
def _lowess_smooth(dep: np.ndarray, x: np.ndarray,
                   frac: float = 0.5, it: int = 3) -> np.ndarray:
    """Robust locally-weighted regression of ``dep`` on ``x`` (monotone x)."""
    if _HAS_LOWESS:
        res = _lowess(dep, x, frac=frac, it=it, return_sorted=True)
        return np.asarray(res[:, 1], dtype=np.float64)
    # fallback: moving-median + moving-mean cascade
    k = max(3, int((len(x) * frac) // 2) * 2 - 1)
    return _moving_median_robust(dep, k)


def _moving_median_robust(sig: np.ndarray, k: int) -> np.ndarray:
    from scipy.ndimage import median_filter, uniform_filter1d
    k = max(k, 3)
    out = median_filter(sig, size=k, mode="nearest")
    # one light moving-average pass to remove the median's stairs
    return uniform_filter1d(out, size=max(3, k // 2), mode="nearest")


# ---------------------------------------------------------------------------
# 2b. Sustained-segment-jump rejection (Hampel blind spot)
# ---------------------------------------------------------------------------
def _sustained_jump_mask(xc: np.ndarray, yc: np.ndarray, t: np.ndarray,
                         frac: float = 0.1, k: float = 6.0,
                         min_run: int = 10, flat_thr: float = 0.4) -> np.ndarray:
    """Flag contiguous segments that are a *sustained, flat* offset from a local
    trend — the "tracer rode a neighbouring curve for N points" artifact.

    Strategy: **transition-boundary pairing** (CUSUM-style change-point detection).

    A sustained jump manifests as two *discontinuities* in first-differences:
    a large step-UP at the start and a large step-DOWN at the end (or vice
    versa).  Between them, first-differences are *normal* (the shifted curve
    is smooth).  We detect boundary pairs, then flag the whole segment.

    Why not one-sided medians: inside a long jump both backward and forward
    medians are contaminated (majority-jump window), so only the first/last
    ~half-window points have detectable residuals — too few for the run-length
    criterion.  Transition boundaries are unambiguous.

    Why not Hampel: Hampel compares each point to a *local* median.  Inside a
    sustained jump the whole neighbourhood is shifted, so the local median is
    also shifted and the spike is invisible.

    A genuine *kink* (yield point, creep stage transition) produces only ONE
    boundary (a change in slope), not a PAIR of opposite-sign boundaries.
    The pairing criterion (step-UP followed by step-DOWN ≥ min_run later)
    excludes kinks.  We additionally require flatness (offset CV < flat_thr)
    within the segment [residual control chart / CUSUM logic].
    """
    n = len(xc)
    good = np.ones(n, dtype=bool)
    if n < min_run + 4:
        return good

    # ---- detect transition boundaries via first-differences ----
    # A jump creates a large |Δy| at start and end; genuine kinks don't have
    # the paired opposite-sign pattern.
    dy = np.diff(yc)
    dx = np.diff(xc)
    step = np.hypot(dx, dy)

    # robust threshold on step magnitudes (MAD-based)
    med_s = float(np.median(step))
    mad_s = float(np.median(np.abs(step - med_s)) * 1.4826)
    if mad_s <= 1e-12:
        return good
    step_thr = k * mad_s  # same k used for Hampel (typically 6)

    # boundary candidates: large step magnitudes
    step_flag = step > step_thr

    # ---- pair step-up and step-down boundaries ----
    # For each boundary at position i (between point i and i+1), classify as
    # step-UP (y increases) or step-DOWN (y decreases).
    step_sign = np.sign(dy)  # +1 = up, -1 = down

    # find all boundary positions; skip edges (first/last min_run points)
    # to avoid spurious boundaries from log-transform / curve start/end
    boundary_idx = np.where(step_flag)[0]
    boundary_idx = boundary_idx[(boundary_idx >= min_run) & (boundary_idx < n - min_run - 1)]

    i = 0
    while i < len(boundary_idx):
        b1 = boundary_idx[i]
        s1 = step_sign[b1]
        # look for a matching opposite-sign boundary at least min_run later
        accepted = False
        for j in range(i + 1, len(boundary_idx)):
            b2 = boundary_idx[j]
            if (b2 - b1) < min_run:
                continue
            s2 = step_sign[b2]
            if s2 == -s1:
                # matched pair: flag segment [b1+1, b2]
                seg_start = b1 + 1
                seg_end = b2 + 1  # +1 because diff is between i and i+1
                seg = yc[seg_start:seg_end]
                # flatness check: offset must be roughly constant
                seg_ref = np.concatenate([yc[max(0, seg_start - min_run):seg_start],
                                           yc[seg_end:min(n, seg_end + min_run)]])
                if len(seg_ref) >= 2 and len(seg) >= 2:
                    offset = float(np.mean(seg)) - float(np.mean(seg_ref))
                    seg_cv = float(np.std(seg - offset) / (abs(offset) + 1e-12))
                else:
                    seg_cv = 999.0  # too short to evaluate → reject
                if seg_cv < flat_thr:
                    # ---- guard: require segment extent >> surrounding extent ----
                    # At a genuine sharp turn (polar loop vertex), the step is
                    # large but the segment is a narrow peak; surrounding steps
                    # are also large (high local curvature).  A jump segment has
                    # flat interior steps that are << boundary steps.
                    seg_steps = step[seg_start:seg_end - 1] if seg_end - seg_start > 1 else np.array([0.0])
                    ref_steps = np.concatenate([
                        step[max(0, seg_start - 5):seg_start],
                        step[seg_end:min(n - 1, seg_end + 5)],
                    ])
                    if len(seg_steps) == 0 or len(ref_steps) == 0:
                        continue
                    interior_flat = float(np.median(seg_steps))
                    normal_step = float(np.median(ref_steps))
                    # For a jump: interior is flat (near-normal steps), boundary is huge.
                    # For a sharp turn: interior is also high (curved), boundary barely above threshold.
                    # So require: boundary step >> interior step
                    boundary_steps = np.array([step[b1], step[b2]])
                    if boundary_steps.min() > 2.0 * (normal_step + interior_flat) or \
                       boundary_steps.min() > 3.0 * interior_flat:
                        good[seg_start:seg_end] = False
                    elif interior_flat < 0.5 * normal_step:
                        # fallback: interior much flatter than surroundings → jump
                        good[seg_start:seg_end] = False
                    accepted = True
                # whether accepted or rejected, this pair is consumed —
                # either the segment is flagged or it's not a real jump.
                # In both cases, advance past the pair so we don't re-match
                # the same boundary.
                break
            elif s2 == s1:
                # same sign: skip, keep looking for opposite
                continue
        if accepted:
            i = j + 1  # skip past both boundaries of the accepted pair
        else:
            i += 1  # no accepted pair found; try next starting boundary

    return good


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def clean_curve_points(
    points,
    x_is_log: bool = False,
    y_is_log: bool = False,
    spike_window: int = 9,
    spike_k: float = 4.0,
    lo_frac: float = 0.1,
    lo_iter: int = 3,
    jump_detect: bool = True,
    jump_k: float = 6.0,
    jump_min_run: int = 10,
    jump_flat_thr: float = 0.4,
    n_out: int | None = None,
    monotone_y: bool = False,
    min_points: int = 5,
) -> "list[tuple[float, float]]":
    """Return cleaned (x, y) data points for redrawing.

    Parameters
    ----------
    points
        Sequence of (x, y) data-coordinate tuples in extraction order.
    x_is_log, y_is_log
        Whether the axis is logarithmic (transform to log10 before cleaning).
    spike_window
        Hampel window size (odd).  Larger = fewer, more isolated spikes removed.
    spike_k
        Hampel rejection threshold in robust-sigma units.  Higher = only very
        extreme spikes removed; lower = more aggressive.
    lo_frac
        LOWESS span (fraction of the curve).  Smaller = follows sharper curve,
        keeps more detail; larger = smoother.

        The default 0.1 is a *light* smoothing that removes pixel-jitter while
        still preserving genuine kinks and stage curvature.  Measured ablation
        (see 曲线_重绘_调研.md): frac>0.2 warps smooth creep/polarization shape;
        frac=0.1 keeps creep nRMSE ~1% of range while removing spikes.
    lo_iter
        Robustness iterations for LOWESS (bisquare reweighting).
    jump_detect
        If True (default), run a second outlier stage that removes *sustained
        flat-offset segments* (tracer riding a neighbouring curve) which the
        per-point Hampel test cannot see.  Set False to disable.
    jump_k
        Jump residual threshold in robust-sigma (MAD) units.  Higher = only very
        large sustained offsets removed.
    jump_min_run
        Minimum length (points) of a sustained offset run to be treated as a
        jump.  Shorter runs are left to LOWESS smoothing.  Larger = only very
        long drift segments removed; smaller = more aggressive.
    jump_flat_thr
        Max within-run residual flatness (std/mean) for a run to count as a
        jump.  A genuine kink yields a *peaked* (non-flat) residual and is kept;
        a jump yields a *flat* (sustained) residual and is removed.  Larger =
        more permissive toward flat runs.
    n_out
        Optional downsampling target length for the output curve; if None the
        cleaned length is returned.
    monotone_y
        If True, project the cleaned, smoothed ``y`` to be non-decreasing in
        the path order (isotonic).  **Only enable when monotonicity is
        physically guaranteed** (e.g. creep strain vs time).  Default False.
    min_points
        Below this input length the points are returned unchanged (too few to
        reject/smooth meaningfully).

    Returns
    -------
    list
        Cleaned points as (x, y) pairs.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return [tuple(float(v) for v in p) for p in pts]
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    # Drop non-finite
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < min_points:
        return [tuple(float(a), float(b)) for a, b in zip(x, y)]

    # Log transform so "a few percent" means the same across the range.
    # Only transform where every value is positive; otherwise keep raw.
    xw, yw = x, y
    if x_is_log and float(x[finite].min()) > 0:
        xw = np.log10(np.where(x > 0, x, x[finite].min()))
    if y_is_log and float(y[finite].min()) > 0:
        yw = np.log10(np.where(y > 0, y, y[finite].min()))

    # ---- spike rejection (per coordinate, iterated once with union mask) ----
    m_x = _hampel_mask(xw, window=spike_window, k=spike_k)
    m_y = _hampel_mask(yw, window=spike_window, k=spike_k)
    keep = finite & m_x & m_y
    if int(keep.sum()) < min_points:
        keep = finite  # over-rejection guard: revert to finite set
    xc, yc = xw[keep], yw[keep]

    # ---- second stage: sustained segment-jump rejection (Hampel blind spot) ----
    if jump_detect and len(xc) >= jump_min_run + 4:
        t0 = np.arange(len(xc), dtype=np.float64) * (1.0 + 1e-9)
        jm = _sustained_jump_mask(xc, yc, t0, frac=lo_frac, k=jump_k,
                                   min_run=jump_min_run, flat_thr=jump_flat_thr)
        if int(jm.sum()) >= min_points:
            xc, yc = xc[jm], yc[jm]
        # over-rejection guard: if it dropped too much, keep the spike-cleaned set
        elif int(jm.sum()) < min_points:
            pass

    # ---- parametric robust smoothing along path order t ----
    n = len(xc)
    if n < min_points:
        return [tuple(float(a), float(b)) for a, b in zip(x[keep], y[keep])]
    t = np.arange(n, dtype=np.float64)
    # t must be strictly increasing & unique for LOWESS
    t = t * (1.0 + 1e-9)
    xs = _lowess_smooth(xc, t, frac=lo_frac, it=lo_iter)
    ys = _lowess_smooth(yc, t, frac=lo_frac, it=lo_iter)

    # ---- optional monotone projection (ONLY when physically guaranteed) ----
    if monotone_y:
        from scipy.optimize import isotonic_regression
        ys = np.asarray(isotonic_regression(ys).x, dtype=np.float64)

    # ---- map back from log space ----
    xr = (10.0 ** xs) if x_is_log and float(x[finite].min()) > 0 else xs
    yr = (10.0 ** ys) if y_is_log and float(y[finite].min()) > 0 else ys

    # ---- optional uniform downsampling ----
    if n_out is not None and n > n_out and n_out >= 2:
        idx = np.linspace(0, n - 1, int(n_out)).round().astype(int)
        xr, yr = xr[idx], yr[idx]

    return [(float(a), float(b)) for a, b in zip(xr, yr)]


# ---------------------------------------------------------------------------
# 3. Keypoint + shape-preserving connection (the "key-point extraction" approach)
# ---------------------------------------------------------------------------
def _rdp_simplify(pts, epsilon):
    """Ramer-Douglas-Peucker polyline simplification (recursive).

    ``pts`` is (N,2) numpy array.  ``epsilon`` is the max perpendicular
    distance (in chain-coordinate units) for a point to be pruned.
    Returns a boolean mask of kept points.
    """
    n = len(pts)
    keep = np.ones(n, dtype=bool)
    if n <= 2:
        return keep
    start, end = 0, n - 1
    stack = [(start, end)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        seg = pts[e] - pts[s]
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len < 1e-12:
            keep[s + 1:e] = False
            continue
        # perpendicular distance of each interior point from line(s->e)
        d = np.zeros(e - s - 1)
        for i in range(s + 1, e):
            p = pts[i] - pts[s]
            cross = abs(seg[0] * p[1] - seg[1] * p[0])
            d[i - s - 1] = cross / seg_len
        mi = int(np.argmax(d))
        maxd = d[mi]
        if maxd <= epsilon:
            keep[s + 1:e] = False
        else:
            idx = s + mi + 1
            stack.append((s, idx))
            stack.append((idx, e))
    return keep


def _curvature_mask(pts, curve_thr, min_sep):
    """Keep points that are curvature local maxima above threshold,
    plus endpoints, plus derivative-sign changes (monotone↔non-monotone).
    Pruned so two kept points are >= min_sep apart."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 3:
        return np.ones(n, dtype=bool)
    ext = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1e-9)
    thr = curve_thr * ext
    # discrete curvature
    v1 = pts[1:] - pts[:-1]
    v2 = pts[2:] - pts[1:-1]
    cross = np.abs(v1[:-1, 0] * v2[:, 1] - v1[:-1, 1] * v2[:, 0])
    denom = np.hypot(v1[:-1, 0], v1[:-1, 1]) * np.hypot(v2[:, 0], v2[:, 1]) + 1e-12
    kappa = np.zeros(n)
    kappa[1:-1] = cross / denom
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    for i in range(1, n - 1):
        if kappa[i] > thr and kappa[i] >= kappa[i - 1] and kappa[i] >= kappa[i + 1]:
            keep[i] = True
    # derivative sign change
    dx = np.diff(pts[:, 0])
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
    slope = np.diff(pts[:, 1]) / dx
    sgn = np.sign(slope)
    for i in range(1, n - 1):
        if sgn[i - 1] != 0 and sgn[i] != 0 and sgn[i - 1] != sgn[i]:
            keep[i] = True
    # prune by spacing
    idx = np.where(keep)[0]
    sel = []
    last = -1
    for i in idx:
        if last < 0:
            sel.append(i)
            last = i
        elif float(np.hypot(*(pts[i] - pts[last]))) >= min_sep * ext:
            sel.append(i)
            last = i
    if 0 not in sel:
        sel.insert(0, 0)
    if n - 1 not in sel:
        sel.append(n - 1)
    sel = sorted(set(sel))
    mask = np.zeros(n, dtype=bool)
    mask[sel] = True
    return mask


def clean_curve_keypoints(
    points,
    x_is_log: bool = False,
    y_is_log: bool = False,
    spike_window: int = 9,
    spike_k: float = 4.0,
    lo_frac: float = 0.1,
    lo_iter: int = 3,
    connect: str = "pchip",
    n_out: int = 300,
    monotone_y: bool = False,
    min_points: int = 5,
) -> "list[tuple[float, float]]":
    """Key-point extraction + shape-preserving connection reconstruction.

    Implements the "extract shape-defining key-points, then connect" paradigm
    observed in curve extraction tools.  Uses the **multi-scale** paradigm
    (Curvature Scale Space, CSS [Mokhtarian & Bober 1995; arXiv 2211.08376]):
    smooth FIRST to suppress noise, then detect features on the smoothed signal
    — this prevents noise-induced false keypoints that geometry-only
    simplifiers (RDP) produce.

    Pipeline:
      1. **Hampel spike rejection** — removes isolated wild points.
      2. **Dense robust pre-smoothing** (LOWESS, bisquare, ``lo_frac``) —
         produces a jitter-free baseline that follows genuine kinks.  This is
         the same smoothing used in ``clean_curve_points``, but here it serves
         as the *feature-detection input*, not the final output.
      3. **Feature extraction on the smoothed curve** — finds:
         (a) local extrema of y (peaks/valleys — material curves have specific
             peaks for polarization, yield for stress-strain);
         (b) high-curvature points (curvature local maxima — kinks, stage
             transitions for creep);
         (c) derivative-sign changes (monotone↔non-monotone transitions);
         (d) endpoints.
         All pruning is scale-adaptive (relative to curve extent).
      4. **Shape-preserving connection** via **PCHIP** (monotone-preserving,
         no overshoot [Fritsch & Butland 1984]) for parametric curves,
         or **Akima** (smooth, low-oscillation).

    Why not RDP: the Ramer-Douglas-Peucker algorithm is a geometry simplifier
    designed for clean polylines; on noisy data it *preserves spikes as
    high-curvature features* — exactly the opposite of denoising.
    Multi-scale CSS (smooth → detect) is the correct noise-robust paradigm
    [Fang & Li, 2016; springerprofessional.de survey].

    Parameters
    ----------
    lo_frac
        LOWESS span for pre-smoothing (default 0.1 — same as
        ``clean_curve_points``).  Used ONLY for feature detection; the final
        output is the PCHIP through keypoints, not the smoothed signal.
    connect
        Interpolation: ``'pchip'`` (monotone-preserving) or ``'akima'``.

    Returns
    -------
    list
        Uniformly sampled output curve at ``n_out`` points.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return [tuple(float(v) for v in p) for p in pts]
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()

    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < min_points:
        return [tuple(float(a), float(b)) for a, b in zip(x, y)]

    # log transform
    xw, yw = x, y
    if x_is_log and float(x[finite].min()) > 0:
        xw = np.log10(np.where(x > 0, x, x[finite].min()))
    if y_is_log and float(y[finite].min()) > 0:
        yw = np.log10(np.where(y > 0, y, y[finite].min()))

    # ---- Stage 1: Hampel spike rejection ----
    m_x = _hampel_mask(xw, window=spike_window, k=spike_k)
    m_y = _hampel_mask(yw, window=spike_window, k=spike_k)
    keep = finite & m_x & m_y
    if int(keep.sum()) < min_points:
        keep = finite
    xc, yc = xw[keep], yw[keep]
    if len(xc) < min_points:
        return [tuple(float(a), float(b)) for a, b in zip(x[keep], y[keep])]

    # ---- Stage 2: dense robust pre-smoothing (for feature detection only) ----
    n = len(xc)
    t = np.arange(n, dtype=np.float64) * (1.0 + 1e-9)
    xs = _lowess_smooth(xc, t, frac=lo_frac, it=lo_iter)
    ys = _lowess_smooth(yc, t, frac=lo_frac, it=lo_iter)

    # ---- Stage 3: feature extraction on the smoothed curve ----
    km = np.zeros(n, dtype=bool)
    km[0] = km[-1] = True  # endpoints

    # (a) local extrema of smoothed y
    for i in range(1, n - 1):
        if (ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1]) or \
           (ys[i] <= ys[i - 1] and ys[i] <= ys[i + 1]):
            km[i] = True

    # (b) high-curvature points (curvature local maxima) — kinks, transitions
    ext = max(float(np.ptp(xs)), float(np.ptp(ys)), 1e-9)
    v1 = np.column_stack([np.diff(xs), np.diff(ys)])
    v2 = np.column_stack([np.diff(xs)[1:], np.diff(ys)[1:]])
    cross = np.abs(v1[:-1, 0] * v2[:, 1] - v1[:-1, 1] * v2[:, 0])
    denom = np.hypot(v1[:-1, 0], v1[:-1, 1]) * np.hypot(v2[:, 0], v2[:, 1]) + 1e-12
    kappa = np.zeros(n)
    kappa[1:-1] = cross / denom
    curv_thr = 0.04 * ext
    for i in range(1, n - 1):
        if kappa[i] > curv_thr and kappa[i] >= kappa[i - 1] and kappa[i] >= kappa[i + 1]:
            km[i] = True

    # (c) derivative-sign changes (monotone ↔ non-monotone)
    dx = np.diff(xs)
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
    slope = np.diff(ys) / dx
    sgn = np.sign(slope)
    for i in range(1, n - 1):
        if sgn[i - 1] != 0 and sgn[i] != 0 and sgn[i - 1] != sgn[i]:
            km[i] = True

    # prune by spacing (min 3 points between keypoints)
    km_idx = np.where(km)[0]
    if len(km_idx) < 2:
        return [(float(xs[i]), float(ys[i])) for i in range(n)]

    sel = [km_idx[0]]
    for i in km_idx[1:]:
        if i - sel[-1] >= 3:
            sel.append(i)
    if sel[-1] != km_idx[-1]:
        sel.append(km_idx[-1])
    km = np.zeros(n, dtype=bool)
    km[sel] = True

    kx, ky = xs[km], ys[km]
    if len(kx) < 2:
        return [(float(xs[i]), float(ys[i])) for i in range(n)]

    # ---- Stage 4: shape-preserving connection ----
    nk = len(kx)
    t_kp = np.linspace(0, 1, nk)
    t_out = np.linspace(0, 1, n_out)
    try:
        if connect == "pchip":
            from scipy.interpolate import PchipInterpolator
            xr = PchipInterpolator(t_kp, kx)(t_out)
            yr = PchipInterpolator(t_kp, ky)(t_out)
        elif connect == "akima":
            from scipy.interpolate import Akima1DInterpolator
            xr = Akima1DInterpolator(t_kp, kx)(t_out)
            yr = Akima1DInterpolator(t_kp, ky)(t_out)
        else:
            xr = np.interp(t_out, t_kp, kx)
            yr = np.interp(t_out, t_kp, ky)
    except Exception:  # noqa: BLE001
        xr = np.interp(t_out, t_kp, kx)
        yr = np.interp(t_out, t_kp, ky)

    if monotone_y:
        from scipy.optimize import isotonic_regression
        yr = np.asarray(isotonic_regression(yr).x, dtype=np.float64)

    xr = (10.0 ** xr) if x_is_log and float(x[finite].min()) > 0 else xr
    yr = (10.0 ** yr) if y_is_log and float(y[finite].min()) > 0 else yr

    return [(float(a), float(b)) for a, b in zip(xr, yr)]