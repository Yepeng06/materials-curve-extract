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

import itertools
import re
from typing import List, Optional, Tuple

import numpy as np

from ..schema import AxisKind, Tick
from ..utils import parse_number_text


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


def _candidates(text: str) -> List[float]:
    """Plausible numeric values for an OCR'd tick text (original parse first).

    matplotlib log-axis labels render decade ticks as 10^k with a real
    superscript (10^0, 10^-1, ...); PP-OCRv4 misreads the superscript in
    several recurring ways (B-5a diagnostics, 12 worst images):

      * '101'/'102'/'103'  -- superscript digit glued to '10'  (10^1..10^3)
      * '100'              -- superscript 0 glued on              (10^0 = 1)
      * '10.0'             -- superscript 0 read as '.0'          (10^0 = 1)
      * '10'               -- superscript lost                    (10^0 = 1
                             or 10^-1 = 0.1 when minus+exp both lost)
      * '10-'              -- exponent lost                       (10^-1..2)
      * '012'/'0-2'        -- broken reads of 10^-2               (0.01)

    The sequence-context disambiguation in :func:`resolve_values` picks
    among these; a genuine '100'/'10' on a linear axis stays itself because
    that keeps the arithmetic progression.
    """
    t = text.strip()
    t = re.sub(r"[\s\u00A0\u2009]+", "", t)
    v = parse_number_text(t)
    cands: List[float] = [v] if v is not None else []
    # leading/trailing-dot and broken reads of the 10N family:
    # '.102'/'102.' (10^2 with a stray dot), '0-1' (10^-1/10^-2 with
    # '1' read as '0')
    t_norm = t.lstrip(".·").rstrip(".·")
    if t_norm != t or t in ("0-1",):
        if t in ("0-1",):
            cands += [0.1, 0.01]
        m0 = _10N.match(t_norm)
        if m0:
            e0 = int(m0.group(1))
            if e0 != 0:
                cands.append(10.0 ** e0)
    if t == "10":
        cands += [1.0, 0.1]
    elif t == "100":
        cands.append(1.0)
    elif t == "10.0":
        cands.append(1.0)
    elif t == "10-":
        cands += [0.1, 0.01]
    elif t in ("012", "0-2"):
        cands.append(0.01)
    m = _10N.match(t)
    if m:
        e = int(m.group(1))
        if e != 0:
            cands.append(10.0 ** e)
    seen: List[float] = []
    for c in cands:
        if not any(abs(c - s) <= 1e-9 * max(1.0, abs(c)) for s in seen):
            seen.append(c)
    return seen


def _seq_score_of(vals: List[Optional[float]],
                    pixels: Optional[List[float]] = None) -> float:
    """Consistency score of a value list (smaller = better; 0 = perfect).

    When pixel positions are available (B-5a), the criterion is the relative
    std of the *px-per-unit* ratio: log axes have a constant px-per-decade,
    linear axes a constant px-per-value-unit.  This couples the value
    sequence to the geometry and rejects 'perfect' geometric progressions
    that do not match the tick spacing (e.g. 3 values of a 4-tick log axis
    with a missing tick misread into a clean [-2,-2] decade run over unequal
    pixel gaps).  Falls back to the value-only _seq_scores criterion; for
    2 values a power-of-ten pair scores 0 (log) and anything else 0.5.
    """
    f = [(float(p), v) for p, v in zip(pixels or [], vals) if v is not None]
    if len(f) < 3:
        return 0.0 if _value_sequence_vote([v for _, v in f]) is AxisKind.LOG else 0.5
    f.sort()
    p = np.asarray([x[0] for x in f], dtype=np.float64)
    v = np.asarray([x[1] for x in f], dtype=np.float64)
    dp = np.diff(p)
    if len(f) >= 3 and float(np.ptp(p)) > 1e-9:
        dv = np.diff(v)
        # a real axis is monotone: alternating signs (e.g. [0.9,-0.9,0.9]
        # from 0.1,1,0.1,1) are not a progression, no matter how 'regular'
        # the |diff| ratios look after abs()
        if (float(np.min(np.abs(dv))) > 1e-12 * max(1.0, float(np.max(np.abs(v))))
                and not (np.any(dv > 0) and np.any(dv < 0))):
            r_lin = dp / np.abs(dv)
            if float(np.mean(r_lin)) > 1e-12:
                lin = float(np.std(r_lin)) / float(np.mean(r_lin))
            else:
                lin = float("inf")
        else:
            lin = float("inf")
    else:
        lin = float("inf")
    pos = v > 0
    if int(pos.sum()) >= 3:
        # adjacent positive ticks sorted by pixel: ldp and ldv correspond
        p_pos = p[pos]
        v_pos = v[pos]
        order = np.argsort(p_pos)
        ldv = np.diff(np.log10(v_pos[order]))
        ldp = np.diff(p_pos[order])
        if (float(np.min(np.abs(ldv))) > 1e-9
                and not (np.any(ldv > 0) and np.any(ldv < 0))):
            r_log = ldp / np.abs(ldv)
            if float(np.mean(r_log)) > 1e-12:
                log = float(np.std(r_log)) / float(np.mean(r_log))
            else:
                log = float("inf")
        else:
            log = float("inf")
        # B-5a note: a decade-gap penalty was tried here but REMOVED -- it
        # broke genuine log axes with a missing tick (img_0073: 1,100,1000
        # reads as 'consistent' as the misread 100,102,103; the 10N
        # preference in resolve_values breaks that tie correctly).
    else:
        log = float("inf")
    s = min(lin, log)
    return s if np.isfinite(s) else 0.5


def resolve_values(ticks: List[Tick]) -> Tuple[List[Optional[float]], bool]:
    """Resolve tick values, re-interpreting the '10N' superscript misread
    family (B-5a): '100'/'10'/'10-'/'012'/'0-2'/'10.0'/'101' etc.

    Every ambiguous tick carries a candidate list (:func:`_candidates`);
    the combination that makes the whole-axis value sequence most consistent
    (arithmetic or geometric, same criterion as before) wins.  A genuine
    '100' on a linear axis is safe: keeping it preserves the progression.

    Returns (values, changed) with values the SAME length as ticks (None
    for valueless ticks).  Preserves the original parse on ties.
    """
    n = len(ticks)
    base: List[Optional[float]] = [None] * n
    cands: List[Optional[List[float]]] = [None] * n
    for i, t in enumerate(ticks):
        c = _candidates(t.text)
        if t.value is None and not c:
            continue
        base[i] = t.value
        cands[i] = c if len(c) > 1 else None

    if sum(1 for v in base if v is not None) < 2:
        return base, False
    amb = [i for i in range(n) if cands[i]]
    if not amb:
        return base, False

    grid = [cands[i] for i in amb]  # type: ignore[list-item]
    total = 1
    for g in grid:
        total *= len(g)
    pixels = [t.pixel for t in ticks]
    n_values = sum(1 for v in base if v is not None)

    def _is_10n(c: float) -> bool:
        if c <= 0:
            return False
        l = float(np.log10(c))
        return abs(l - round(l)) < 1e-6

    def _better(a: Tuple[float, int, int], b: Tuple[float, int, int]) -> bool:
        # (score, n_10n_rereads, n_diffs): score dominates; within EPS the
        # combination that uses more 10^N re-resolutions wins -- pixel noise
        # of ~2% makes '102' vs 10^2 nearly tied, and the 10N reading is
        # the matplotlib-log reality.  A perfect 0.0 tie keeps the original
        # parse (e.g. the 2-tick power-of-ten pair '0.1','10').
        sa, pa, na = a
        sb, pb, nb = b
        if sa < sb - 0.01:
            return True
        if sa > sb + 0.01:
            return False
        # within EPS
        if sa == 0.0 and sb == 0.0:
            # both progressions perfect: glued superscripts make e.g.
            # 100,101,102,103 perfectly arithmetic AND geometric -- the
            # 10N re-resolution is the matplotlib-log reality.  With 2
            # values there is no sequence information at all, so keep the
            # original parse.
            if n_values >= 3 and pa != pb:
                return pa > pb
            return na < nb
        if sa == 0.0:
            return True
        if sb == 0.0:
            return False
        if pa != pb:
            return pa > pb
        return na < nb

    if total <= 256:
        best: Optional[Tuple[Tuple[float, int, int], List[Optional[float]]]] = None
        for combo in itertools.product(*grid):
            vals = list(base)
            n10 = 0
            for i, c in zip(amb, combo):
                vals[i] = c
                if c != base[i] and _is_10n(c):
                    n10 += 1
            s = _seq_score_of(vals, pixels)
            ndiff = sum(1 for i, c in zip(amb, combo) if c != base[i])
            key = (s, n10, ndiff)
            if best is None or _better(key, best[0]):
                best = (key, vals)
        assert best is not None
        vals = best[1]
    else:
        # greedy fallback: fix each ambiguous tick one at a time
        vals = list(base)
        for i in amb:
            cur_best: Optional[Tuple[float, float]] = None
            for c in cands[i]:  # type: ignore[union-attr]
                vals[i] = c
                s = _seq_score_of(vals, pixels)
                if cur_best is None or s < cur_best[0]:
                    cur_best = (s, c)
            assert cur_best is not None
            vals[i] = cur_best[1]

    changed = any(vals[i] != base[i] for i in amb)
    return vals, changed


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

