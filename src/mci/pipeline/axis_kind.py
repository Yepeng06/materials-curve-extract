"""Multi-signal coordinate-type judgement (Phase B-3).

Three independent signals vote on whether an axis is linear or log:
  1. tick-value sequence consistency -- arithmetic vs geometric
     progression -- with '10N' superscript-misread re-resolution
     (matplotlib log labels render 10^1 as a superscript glyph that OCR
     glues into '101'; re-interpreting '10N' as 10^N restores the
     geometric progression),
  2. tick-pixel spacing analysis -- minor-tick density, which does NOT
     depend on OCR values (log axes carry ~9 minors per decade, linear
     axes at most 2-5 per major interval),
  3. an external prior (axis-title text hint from Phase B-2, or a
     config-level override).

A decided vote wins (ties fall back to the R^2 double-fit criterion in
coordinate_mapper).  R^2 then only grades the chosen mapping.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

from ..schema import AxisKind, Tick


_10N = re.compile(r"^10([0-9])$")  # '10N' glued superscript (N = exponent)


def _seq_scores(vals: List[float]) -> Optional[Tuple[float, float]]:
    """Consistency scores (lin, log); smaller = more consistent.

    lin: relative std of first differences (arithmetic progression).
    log: relative std of log10 differences (geometric progression).
    Returns None when undecidable (<3 valued ticks).
    """
    v = np.asarray(vals, dtype=np.float64)
    if len(v) < 3:
        return None
    d = np.diff(v)
    lin = float(np.std(d)) / (float(np.mean(np.abs(d))) + 1e-12)
    pos = v[v > 0]
    if len(pos) >= 3 and float(np.ptp(pos)) > 1e-9:
        ld = np.diff(np.log10(pos))
        log = float(np.std(ld)) / (float(np.mean(np.abs(ld))) + 1e-12)
    else:
        log = float("inf")
    return lin, log


def resolve_values(ticks: List[Tick]) -> Tuple[List[Optional[float]], bool]:
    """Resolve tick values, re-interpreting glued '10N' superscripts.

    Returns (values, re_resolved) where values has the SAME length as
    ticks (None entries for valueless ticks -- callers zip them back onto
    the ticks).  When every '10N' text re-read as 10^N makes the value
    sequence clearly MORE consistent (geometric or arithmetic), the
    re-resolved values are used.  A genuine '100' on a linear axis is
    safe: re-reading it as 1 destroys the sequence.
    """
    n = len(ticks)
    base: List[Optional[float]] = [None] * n
    alt: List[Optional[float]] = [None] * n
    for i, t in enumerate(ticks):
        if t.value is None:
            continue
        base[i] = t.value
        m = _10N.match(t.text.strip())
        alt[i] = (10.0 ** int(m.group(1))) if m else t.value

    base_f = [v for v in base if v is not None]
    alt_f = [v for v in alt if v is not None]
    if alt_f == base_f or len(base_f) < 2:
        return base, False

    if len(base_f) == 2:
        # 2 ticks: accept the re-resolution when it produces a
        # power-of-ten pair (e.g. '101' -> 10^1 -> [10, 0.1]) while the
        # raw values are not one
        if (_value_sequence_vote(alt_f) is AxisKind.LOG
                and _value_sequence_vote(base_f) is not AxisKind.LOG):
            return alt, True
        return base, False

    s0 = _seq_scores(base_f)
    s1 = _seq_scores(alt_f)
    if s0 is None or s1 is None:
        return base, False
    best0 = min(s0)
    best1 = min(s1)
    # accept when the re-resolved sequence is near-perfect AND clearly
    # better: either the raw sequence is not near-perfect at all, or it is
    # ambiguous (both progressions score ~0, e.g. glued '100/101/102/103'
    # which is trivially arithmetic AND geometric) while the re-resolved
    # one is clearly typed (one score far below the other)
    if best1 < 0.05:
        if best0 >= 0.05 or (abs(s0[0] - s0[1]) < 0.02
                             and abs(s1[0] - s1[1]) > 0.1):
            return alt, True
    return base, False


def _majority_consistency(vals: List[float]) -> Optional[AxisKind]:
    """Tolerant vote: a majority (>=50%) of adjacent gaps/ratios agree.

    Handles 1-2 misread tick values that would otherwise poison the strict
    sequence check (e.g. '200' read as '20', or a duplicated '0'): the
    dominant gap (linear) or dominant log10 ratio (log) wins.
    """
    v = sorted(vals)
    if len(v) < 4:
        return None
    d = np.diff(v)
    best: Tuple[AxisKind, float] = (AxisKind.LINEAR, 0.0)
    med_d = float(np.median(d))
    if abs(med_d) > 1e-9:
        frac = float(np.mean(np.abs(d - med_d) <= 0.1 * abs(med_d)))
        best = (AxisKind.LINEAR, frac)
    pos = np.array([x for x in v if x > 0])
    if len(pos) >= 4:
        r = np.diff(np.log10(pos))
        med_r = float(np.median(r))
        if abs(med_r) > 1e-9:
            frac = float(np.mean(np.abs(r - med_r) <= 0.1 * abs(med_r)))
            if frac > best[1]:
                best = (AxisKind.LOG, frac)
    return best[0] if best[1] >= 0.5 else None


def _value_sequence_vote(vals: List[float]) -> Optional[AxisKind]:
    """Vote from value-sequence consistency (None = abstain)."""
    if len(vals) == 2:
        # 2 ticks: any model interpolates perfectly, but a pair of power-of-ten
        # values with a power-of-ten ratio (10^n, n>=1) is a strong log signal
        # (matplotlib log axes label decades as 10^k; linear axes almost never
        # pick such a pair).  [1, 10] on a linear axis is the rare false case.
        v1, v2 = sorted(float(x) for x in vals)
        if v1 > 0 and v2 > 0:
            r = v2 / v1
            lr = float(np.log10(r))
            if abs(lr - round(lr)) < 1e-6 and abs(lr) >= 1.0:
                l1, l2 = float(np.log10(v1)), float(np.log10(v2))
                if abs(l1 - round(l1)) < 1e-4 and abs(l2 - round(l2)) < 1e-4:
                    return AxisKind.LOG
        return None
    s = _seq_scores(vals)
    if s is None:
        return None
    lin, log = s
    if min(lin, log) > 0.1:
        # neither strict progression holds: tolerate a minority of misreads
        return _majority_consistency(vals)
    if log < lin:
        return AxisKind.LOG
    if lin < log:
        return AxisKind.LINEAR
    return None


def _pixel_spacing_vote(ticks: List[Tick]) -> Optional[AxisKind]:
    """Vote from tick-pixel spacing (minor-tick density).

    Uses every tick position (marks AND label-centre fallbacks).  A
    single dominant spacing (majors only) is undecidable -- both axis
    kinds place major ticks evenly.  A spacing ratio >= 8 (log axes: 9-10
    minors per decade) votes log; a ratio <= 5 (linear axes: 2-5 equal
    subdivisions) votes linear; anything in between abstains.
    """
    # use valued ticks to define the axis range, then collect every tick
    # position (marks + label centres) inside that range, de-duplicated
    # (marks and label centres of the same tick differ by < 1 px)
    valued_px = sorted(t.pixel for t in ticks if t.value is not None)
    if len(valued_px) < 2:
        return None
    lo, hi = valued_px[0], valued_px[-1]
    all_px = np.asarray(
        sorted({round(t.pixel, 1) for t in ticks if lo - 1.0 <= t.pixel <= hi + 1.0}),
        dtype=np.float64,
    )
    if len(all_px) < 5:
        return None
    d = np.diff(all_px)
    d = d[d > 1e-6]
    if len(d) < 4:
        return None
    d_min, d_max = float(d.min()), float(d.max())
    if d_max <= 1e-9:
        return None
    ratio = d_max / d_min
    # count of the largest spacing (major interval) vs total
    n_major = int(np.sum(d > 0.8 * d_max))
    if ratio < 1.5:
        return None  # single uniform spacing: majors-only (undecidable)
    if ratio >= 8.0 and n_major >= 2:
        return AxisKind.LOG
    if ratio <= 5.0 and n_major >= 2 and d_min > 0.15 * d_max:
        return AxisKind.LINEAR
    return None


def judge_axis_kind(
    ticks: List[Tick], kind_hint: str = "auto",
) -> Tuple[Optional[AxisKind], List[str]]:
    """Multi-signal vote; returns (kind or None, evidence list)."""
    vals, reread = resolve_values(ticks)
    valued = [v for v in vals if v is not None]
    evidence: List[str] = []
    if reread:
        evidence.append("10N superscript re-resolved")

    votes: List[Tuple[AxisKind, float]] = []
    sv = _value_sequence_vote(valued)
    if sv is not None:
        votes.append((sv, 1.0))
        evidence.append(f"value-sequence -> {sv.value}")
    pv = _pixel_spacing_vote(ticks)
    if pv is not None:
        votes.append((pv, 0.8))
        evidence.append(f"pixel-spacing -> {pv.value}")
    if kind_hint == "log":
        votes.append((AxisKind.LOG, 2.0))
        evidence.append("external prior -> log")
    elif kind_hint == "linear":
        votes.append((AxisKind.LINEAR, 2.0))
        evidence.append("external prior -> linear")

    log_w = sum(w for k, w in votes if k is AxisKind.LOG)
    lin_w = sum(w for k, w in votes if k is AxisKind.LINEAR)
    if log_w != lin_w:
        kind = AxisKind.LOG if log_w > lin_w else AxisKind.LINEAR
        evidence.append(f"vote {log_w}:{lin_w} -> {kind.value}")
        return kind, evidence
    return None, evidence

