"""Curve extraction — classical-CV baseline (single curve).

Pipeline inside the plot area:
  1. binarize (Otsu);
  2. remove grid lines — a grid line is a long straight line aligned with a
     detected tick mark (grids always pass through ticks in paper figures),
     or a light/regular parallel family (fallback when ticks are missing);
  3. connected components -> score components (span x sqrt(area) x darkness);
     the best component is the curve (a list so the future multi-curve
     backend just keeps the top-K components + legend matching);
  4. skeletonize -> trace the ordered pixel chain with direction continuity
     (survives grid crossings); if the trace is not sane, fall back to the
     per-column median with gap interpolation;
  5. map pixels to data coordinates via the fitted axes.

This module's ``segment`` step is the replacement point for the U-Net
segmenter (see base.CurveSegmenter).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ..schema import AxisSpec, ChartStructure, Curve, CurveExtractionError
from ..utils import dedupe_consecutive, downsample_chain, ink_mask


# ---------------------------------------------------------------------------
# Grid-line removal
# ---------------------------------------------------------------------------
def _remove_gridlines(ink: np.ndarray, image_bgr: np.ndarray,
                      structure: ChartStructure, cfg: Dict) -> np.ndarray:
    """Remove grid lines while keeping the curve.

    Two complementary mechanisms:

    1. Profile scan: a grid row/column covers > 45% of the plot width/height
       (dashed grids included, since the dashes cover ~60% of the run) and is
       aligned with a detected tick mark.  Curves are 1-4 px thin and can
       never reach 45% coverage, so this is safe.
    2. Hough line families (see below) for grids without detectable ticks.

    Hough segments of the *same* line are first merged by intercept into one
    cluster, so a long straight curve segment (power-law on log-log,
    steady-state creep) is never mistaken for several lines.
    """
    h, w = ink.shape
    x0, y0, x1, y1 = structure.plot_bbox
    y_ticks_local = [py - y0 for py in structure.y_ticks_px if y0 <= py <= y1]
    x_ticks_local = [px - x0 for px in structure.x_ticks_px if x0 <= px <= x1]

    grid_mask = np.zeros_like(ink)
    # ---- mechanism 1: profile + tick alignment ----
    # Remove only rows/columns whose coverage is > 45% within +-1 px of a
    # tick: the grid line itself is 1-2 px thick, so this keeps the removed
    # band narrow (1-3 px) and the closing below can bridge the gaps in the
    # curve where it crossed the grid.
    row_cov = ink.mean(axis=1).astype(np.float64)
    col_cov = ink.mean(axis=0).astype(np.float64)
    for ty in y_ticks_local:
        for r in range(max(0, int(ty) - 1), min(h, int(ty) + 2)):
            if row_cov[r] > 0.45:
                grid_mask[r, :] = 1
    for tx in x_ticks_local:
        for c in range(max(0, int(tx) - 1), min(w, int(tx) + 2)):
            if col_cov[c] > 0.45:
                grid_mask[:, c] = 1

    # ---- mechanism 2: Hough families ----
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    min_len = int(min(h, w) * 0.45)
    lines = cv2.HoughLinesP(
        ink * 255, 1, np.pi / 180,
        threshold=min_len // 2,
        minLineLength=min_len,
        maxLineGap=24,
    )
    if lines is not None:
        h_lines, v_lines = [], []
        for x1_, y1_, x2_, y2_ in lines[:, 0]:
            angle = abs(np.arctan2(y2_ - y1_, x2_ - x1_)) if abs(x2_ - x1_) > 1e-6 else np.pi / 2
            if angle < np.deg2rad(8):
                h_lines.append((int(x1_), int(y1_), int(x2_), int(y2_)))
            elif angle > np.deg2rad(82):
                v_lines.append((int(x1_), int(y1_), int(x2_), int(y2_)))

        def _clusters(lines_xy, horizontal: bool):
            def key(l):
                return (l[1] + l[3]) / 2.0 if horizontal else (l[0] + l[2]) / 2.0

            clusters: List[List] = []
            for l in lines_xy:
                k = key(l)
                for c in clusters:
                    if abs(k - key(c[0])) < 5:
                        c.append(l)
                        break
                else:
                    clusters.append([l])
            return clusters

        def _cluster_key(cluster, horizontal: bool) -> float:
            return float(np.median([(l[1] + l[3]) / 2 if horizontal else (l[0] + l[2]) / 2
                                    for l in cluster]))

        def _cluster_gray(cluster) -> float:
            vals = []
            for x1_, y1_, x2_, y2_ in cluster:
                n = max(abs(x2_ - x1_), abs(y2_ - y1_))
                for i in range(0, n + 1, max(1, n // 24)):
                    x = int(round(x1_ + (x2_ - x1_) * i / n))
                    y = int(round(y1_ + (y2_ - y1_) * i / n))
                    if 0 <= x < w and 0 <= y < h:
                        vals.append(int(gray[y, x]))
            return float(np.mean(vals)) if vals else 255.0

        def _tick_aligned(cluster, horizontal: bool) -> bool:
            k = _cluster_key(cluster, horizontal)
            ticks = y_ticks_local if horizontal else x_ticks_local
            return any(abs(k - t) <= 3 for t in ticks)

        def _to_remove(clusters, horizontal: bool) -> List:
            if not clusters:
                return []
            if len(clusters) < 2:
                return list(clusters) if _tick_aligned(clusters[0], horizontal) else []
            keys = sorted(_cluster_key(c, horizontal) for c in clusters)
            grays = [_cluster_gray(c) for c in clusters]
            gaps = np.diff(np.asarray(keys, dtype=np.float64))
            all_light = all(g > 150 for g in grays)
            regular = len(gaps) >= 2 and float(gaps.max() / max(gaps.min(), 1e-6)) < 2.5
            out = []
            for i, c in enumerate(clusters):
                if _tick_aligned(c, horizontal):
                    out.append(c)
                elif all_light:
                    out.append(c)
                elif regular and i != int(np.argmin(grays)):
                    out.append(c)
            return out

        for c in _to_remove(_clusters(h_lines, True), True):
            x1_, y1_, x2_, y2_ = c[0]
            cv2.line(grid_mask, (x1_, y1_), (x2_, y2_), 1, 2)
        for c in _to_remove(_clusters(v_lines, False), False):
            x1_, y1_, x2_, y2_ = c[0]
            cv2.line(grid_mask, (x1_, y1_), (x2_, y2_), 1, 2)

    out = ink.copy()
    out[grid_mask > 0] = 0
    # reconnect the curve across the removed grid crossings: gaps are 1-3 px
    # (grid line thickness); a 7 px kernel bridges them without fusing
    # unrelated components (large kernels would create one big blob)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k2)
    return out


def _bridge_dash_gaps(mask: np.ndarray, max_gap: int = 48) -> np.ndarray:
    """Reconnect dashed curve fragments by bridging aligned skeleton endpoints.

    Each dash fragment has two endpoints whose outward directions point along
    the curve; when two endpoints face each other across a small gap with
    consistent direction, they are connected with a 2 px line.  Only
    endpoints of *different* components are bridged (no loops).
    """
    if mask.sum() < 4:
        return mask
    n, labels = cv2.connectedComponents(mask, 8)
    skel = skeletonize(mask.astype(bool)).astype(np.uint8)
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if len(pts) < 4:
        return mask

    def nbrs(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts]

    ends = [p for p in pts if len(nbrs(p)) == 1]
    out = mask.copy()
    if len(ends) < 2:
        return out

    # candidate pairs: different components, gap within range, directions
    # facing each other along the gap
    cand = []  # (dist, e_idx, e2_idx)
    for i, e in enumerate(ends):
        n1 = nbrs(e)[0]
        d1 = np.array([e[0] - n1[0], e[1] - n1[1]], dtype=np.float64)
        l1 = np.hypot(*d1)
        if l1 < 1e-6:
            continue
        d1 = d1 / l1
        for j in range(i + 1, len(ends)):
            e2 = ends[j]
            if labels[e[1], e[0]] == labels[e2[1], e2[0]]:
                continue
            gap = np.array([e2[0] - e[0], e2[1] - e[1]], dtype=np.float64)
            d = np.hypot(*gap)
            if d < 3 or d > max_gap:
                continue
            gap_dir = gap / d
            n2 = nbrs(e2)[0]
            d2 = np.array([e2[0] - n2[0], e2[1] - n2[1]], dtype=np.float64)
            l2 = np.hypot(*d2)
            if l2 < 1e-6:
                continue
            d2 = d2 / l2
            if float(d1 @ gap_dir) > 0.5 and float(d2 @ (-gap_dir)) > 0.5:
                cand.append((d, i, j))
    cand.sort(key=lambda t: t[0])

    # greedy assignment: each endpoint bridges at most once; prefer the
    # shortest mutual pair
    used: set = set()
    for dist, i, j in cand:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        cv2.line(out, ends[i], ends[j], 1, 2)
    return out


# ---------------------------------------------------------------------------
# Chain tracing
# ---------------------------------------------------------------------------
def _trace_chain(skel: np.ndarray, prob: Optional[np.ndarray] = None,
                 lookahead: int = 10, prob_weight: float = 0.35,
                 active: Optional[np.ndarray] = None,
                 ) -> Optional[List[Tuple[int, int]]]:
    """Walk the skeleton from the leftmost endpoint with direction continuity.

    At junctions (grid crossings, curve crossings / approach zones) the
    neighbour that continues the current direction is preferred, so the
    walk follows the curve instead of turning onto a crossing line.

    P1b: when ``prob`` (this channel's plot-local probability map) and
    ``active`` (approach-zone mask) are given, junction candidates at
    positions inside ``active`` are additionally scored by the accumulated
    probability along a short lookahead walk of their branch.  Outside
    approach zones the pure direction score is kept (a naive global
    probability term regresses normal charts: 0.7870 -> 0.7759 on val).
    """
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if not pts:
        return None

    def nbrs(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0) and (x + dx, y + dy) in pts]

    deg = {p: len(nbrs(p)) for p in pts}
    endpoints = [p for p in pts if deg[p] == 1]
    start = min(endpoints, key=lambda p: (p[0], p[1])) if endpoints else min(pts, key=lambda p: (p[0], p[1]))
    end = None
    if len(endpoints) >= 2:
        end = min((e for e in endpoints if e != start), key=lambda p: (p[0], p[1]))

    def _walk_score(n0, dir0):
        """Direction score + (inside approach zones) normalized lookahead
        probability sum for the branch starting at candidate n0."""
        # direction part (same as baseline)
        dx, dy = n0[0] - cur[0], n0[1] - cur[1]
        dscore = -(dx * dir0[0] + dy * dir0[1])
        if prob is None or active is None or not active[cur[1], cur[0]]:
            return dscore
        # lookahead walk along the skeleton from n0 (direction continuity)
        acc, cnt = 0.0, 0
        prev_p, p = cur, n0
        for _ in range(lookahead):
            acc += float(prob[p[1], p[0]])
            cnt += 1
            nxt = [m for m in nbrs(p) if m != prev_p]
            if not nxt:
                break
            d = (p[0] - prev_p[0], p[1] - prev_p[1])
            nxt.sort(key=lambda m: (-((m[0] - p[0]) * d[0] + (m[1] - p[1]) * d[1]), m[0], m[1]))
            prev_p, p = p, nxt[0]
        return dscore + prob_weight * (acc / max(cnt, 1))

    path: List[Tuple[int, int]] = []
    cur, prev = start, None
    while True:
        path.append(cur)
        if cur == end:
            break
        cand = [n for n in nbrs(cur) if n != prev and n not in path]
        if not cand:
            cand = [n for n in nbrs(cur) if n != prev]
        if not cand:
            break
        if prev is not None:
            dir0 = (cur[0] - prev[0], cur[1] - prev[1])
            cand.sort(key=lambda n: -_walk_score(n, dir0))
        else:
            cand.sort(key=lambda n: (-n[0], n[1]))
        prev, cur = cur, cand[0]
        if len(path) > len(pts) + 2:
            break

    if len(path) < 3 or len(path) < 0.3 * len(pts):
        return None
    return path


def _column_median(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Median ink row per column, with linear gap interpolation inside the
    curve's x-span. Robust to junctions / loops."""
    h, w = mask.shape
    med = [None] * w
    for x in range(w):
        ys = np.nonzero(mask[:, x])[0]
        if len(ys):
            med[x] = int(np.median(ys))
    xs_ok = [x for x, v in enumerate(med) if v is not None]
    if not xs_ok:
        return []
    if len(xs_ok) == 1:
        return [(xs_ok[0], med[xs_ok[0]])]
    ys_ok = [med[x] for x in xs_ok]
    filled = np.interp(np.arange(w), np.asarray(xs_ok), np.asarray(ys_ok))
    return [(x, int(round(filled[x]))) for x in range(xs_ok[0], xs_ok[-1] + 1)]


def _column_centroid(prob: np.ndarray, thr: float = 0.3) -> List[Tuple[int, float]]:
    """Sub-pixel curve center per column (probability-weighted mean of y).

    Used by the U-Net path when the skeleton trace is not available; note
    that per-column centroids are biased on steep segments (the probability
    band in a column is elongated along the curve), so the trace+refine path
    in ``extract_curves`` is preferred.
    """
    h, w = prob.shape
    ys = np.arange(h, dtype=np.float64)
    xs_ok: List[int] = []
    ys_c: List[float] = []
    for x in range(w):
        col = prob[:, x]
        m = col > thr
        if m.sum() >= 2:
            ys_c.append(float((col[m] * ys[m]).sum() / col[m].sum()))
            xs_ok.append(x)
    if not xs_ok:
        return []
    if len(xs_ok) == 1:
        return [(xs_ok[0], ys_c[0])]
    filled = np.interp(np.arange(xs_ok[0], xs_ok[-1] + 1),
                       np.asarray(xs_ok), np.asarray(ys_c))
    start = xs_ok[0]
    return [(int(start + i), float(v)) for i, v in enumerate(filled)]


def _refine_chain(prob: np.ndarray, chain: List[Tuple[int, int]],
                  radius: int = 6, skip_mask: Optional[np.ndarray] = None,
                  ) -> List[Tuple[float, float]]:
    """Sub-pixel refinement: local probability-weighted centroid around each
    skeleton point.  A small window keeps the centroid on the curve even on
    steep segments (unlike per-column centroids).

    P1: when ``skip_mask`` is given, skeleton points inside it (approach
    zones where another channel also fires) keep their integer skeleton
    position -- the window centroid there is pulled toward the other curve
    (measured: disabling refine moves 7 curves out of the 1-2% bucket,
    recall 0.7352 -> 0.7481)."""
    h, w = prob.shape
    out: List[Tuple[float, float]] = []
    for x, y in chain:
        if skip_mask is not None and skip_mask[y, x]:
            out.append((float(x), float(y)))
            continue
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        patch = prob[y0:y1, x0:x1]
        if patch.max() < 0.3:
            out.append((float(x), float(y)))
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64),
                             np.arange(y0, y1, dtype=np.float64))
        wsum = patch.sum()
        out.append((float((patch * xs).sum() / wsum),
                    float((patch * ys).sum() / wsum)))
    return out


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------
def _filter_mask_fragments(mask01: np.ndarray, plot_w: int, plot_h: int) -> np.ndarray:
    """Drop non-curve mask fragments that corrupt tracing/centroid fallbacks.

    Two fragment classes are removed (kept: the genuine curve):
      (a) thin strips hugging the plot border that span the full plot edge
          (frame / grid remnants — the CV path's historical rule);
      (b) thin components that are short in BOTH directions (title text,
          legend glyphs, dust): a real curve portion is either long in x
          (horizontal segments) or tall in y (steep segments) — or thick.
    Small specks (area < 8 or < 2% of the largest component) are dropped
    too.  U-Net masks are usually solid for dashed curves (the network
    learns dash completion from solid GT masks), so dropping isolated thin
    fragments does not break the dash case.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, 8)
    if n <= 1:
        return mask01
    areas = [stats[i][4] for i in range(1, n)]
    max_area = max(areas)
    out = np.zeros_like(mask01)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < max(8, 0.02 * max_area):
            continue  # noise specks
        thin = (h <= 8 or w <= 8)
        # (a) full-edge frame strips (historical CV rule)
        if thin and w >= 0.85 * plot_w and (y <= 0.08 * plot_h or y + h >= plot_h - 0.08 * plot_h):
            continue
        if thin and h >= 0.85 * plot_h and (x <= 0.05 * plot_w or x + w >= plot_w - 0.05 * plot_w):
            continue
        # (b) thin fragments short in both directions
        if thin and w < 0.5 * plot_w and h < 0.5 * plot_h:
            continue
        out[labels == i] = 1
    return out


def _select_curve_mask_cv(image_bgr: np.ndarray, structure: ChartStructure,
                          cfg: Dict) -> np.ndarray:
    """CV path: binarize + grid removal + dash bridging + component scoring."""
    from .segmenter import cv_plot_mask

    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    ink = cv_plot_mask(image_bgr, structure, cfg)
    ink = _filter_mask_fragments(ink, plot_w, plot_h)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    min_area = int(cfg.get("min_curve_area", 60))
    comps = []
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        # reject axis/frame remnant lines: thin components spanning the crop
        # edge-to-edge (the frame lives on the bbox border); a genuine curve
        # is thick and spans the plot in *both* directions
        span_x = w / plot_w
        span_y = h / plot_h
        thin = (h <= 8 or w <= 8)
        if thin and span_x >= 0.85 and (y <= 0.08 * plot_h or y + h >= plot_h - 0.08 * plot_h):
            continue  # horizontal frame / grid remnant near top or bottom
        if thin and span_y >= 0.85 and (x <= 0.05 * plot_w or x + w >= plot_w - 0.05 * plot_w):
            continue  # vertical frame / grid remnant near left or right
        span = span_x  # fraction of plot width covered
        # prefer darker components (the curve is darker than light grids)
        px = rgb[labels == i]
        mean_gray = float(np.mean(px, axis=1).mean()) if len(px) else 255.0
        darkness = float(np.clip((255.0 - mean_gray) / 128.0, 0.3, 2.0))
        score = span * np.sqrt(area) * darkness
        comps.append((score, i, area, span))
    if not comps:
        raise CurveExtractionError("no curve component found")

    comps.sort(key=lambda c: -c[0])
    return (labels == comps[0][1]).astype(np.uint8)


def extract_curves(
    image_bgr: np.ndarray,
    structure: ChartStructure,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    cfg: Optional[Dict] = None,
    segmenter: Optional["object"] = None,
) -> List[Curve]:
    """Extract curves from the plot region.

    ``segmenter``: None -> classical CV pipeline; otherwise an object with
    ``segment_full(image_bgr) -> 0/1 full-image mask`` (e.g. UNetSegmenter).
    The skeletonize / trace / coordinate mapping stages are shared.
    """
    cfg = cfg or {}
    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    if plot_w < 20 or plot_h < 20:
        raise CurveExtractionError("plot area too small")

    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]

    if segmenter is not None:
        # U-Net path: skeleton trace (topology-safe on steep segments) then
        # sub-pixel refinement with the probability map; column-centroid
        # fallback when the trace is not available.
        prob_fn = getattr(segmenter, "prob_full", None)
        if prob_fn is not None:
            p = prob_fn(image_bgr)
            region = p[y0 : y1 + 1, x0 : x1 + 1]
            if region.max() < 0.5:
                raise CurveExtractionError("segmentation found no curve in plot region")
            mask01 = (region > 0.5).astype(np.uint8)
            # drop title/frame/dust fragments (see _filter_mask_fragments):
            # they corrupt skeleton tracing (trace start on a fragment) and
            # the per-column centroid fallback (fragment pixels averaged in)
            mask01 = _filter_mask_fragments(mask01, plot_w, plot_h)
            if mask01.sum() < 16:
                raise CurveExtractionError("segmentation found no curve in plot region")
            skel = skeletonize(mask01.astype(bool)).astype(np.uint8)
            chain = _trace_chain(skel)
            if chain is not None:
                xs = [c[0] for c in chain]
                if (max(xs) - min(xs) + 1) < 0.7 * plot_w:
                    chain = None
            if chain is not None:
                chain = _refine_chain(region, chain)
            else:
                chain = _column_centroid(region)
            bias_y = float(cfg.get("multi_refine_bias_y", 0.0))
            if bias_y:
                chain = [(px, py + bias_y) for px, py in chain]
            chain = downsample_chain(chain, int(cfg.get("max_points", 2000)))
            points = [(x_axis.pixel_to_value(x0 + px), y_axis.pixel_to_value(y0 + py))
                      for px, py in chain]
            pixel_points = [(x0 + int(round(px)), y0 + int(round(py))) for px, py in chain]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            color_px = rgb[mask01 > 0]
            color = tuple(int(c) for c in np.median(color_px, axis=0)) if len(color_px) else (0, 0, 0)
            curve = Curve(name="curve_0", points=points, pixel_points=pixel_points,
                          color=color, legend_label=None)
            return [curve]
        full = segmenter.segment_full(image_bgr)
        ink = full[y0 : y1 + 1, x0 : x1 + 1].copy()
        ink[:2, :] = 0
        ink[-2:, :] = 0
        ink[:, :2] = 0
        ink[:, -2:] = 0
        ink = _bridge_dash_gaps(ink)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        if n <= 1:
            raise CurveExtractionError("segmentation mask is empty in plot region")
        curve_mask = (labels == 1 + int(np.argmax(stats[1:, 4]))).astype(np.uint8)
    else:
        curve_mask = _select_curve_mask_cv(image_bgr, structure, cfg)

    # color (dominant = median of non-white pixels)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    px = rgb[curve_mask > 0]
    color = tuple(int(c) for c in np.median(px, axis=0)) if len(px) else (0, 0, 0)

    # trace with sanity fallback to column median
    skel = skeletonize(curve_mask.astype(bool)).astype(np.uint8)
    chain = _trace_chain(skel)
    if chain is not None:
        xs = [p[0] for p in chain]
        if (max(xs) - min(xs) + 1) < 0.7 * plot_w or len(chain) < 0.5 * int(curve_mask.sum()):
            chain = None
    if chain is None:
        chain = _column_median(curve_mask)
    chain = dedupe_consecutive(chain)
    chain = downsample_chain(chain, int(cfg.get("max_points", 2000)))

    # map to data coordinates
    points = [(x_axis.pixel_to_value(x0 + px), y_axis.pixel_to_value(y0 + py))
              for px, py in chain]
    pixel_points = [(x0 + px, y0 + py) for px, py in chain]

    curve = Curve(
        name="curve_0",
        points=points,
        pixel_points=pixel_points,
        color=color,
        legend_label=None,
    )
    return [curve]

def _truncate_jumps(chain, max_jump: float = 30.0, min_len: int = 40):
    """Cut chain tails where y jumps abruptly and keeps jumping.

    Phase C: when a channel mask fades out (model unsure, e.g. a steep
    tail at prob 0.44), the tracer follows leftover pixels of another
    curve and the chain jumps tens of pixels; the jumped portion is
    worse than useless (it poisons the RMSE), so it is cut.
    """
    if len(chain) < min_len + 2:
        return chain
    xs = np.asarray([p[0] for p in chain], dtype=np.float64)
    ys = np.asarray([p[1] for p in chain], dtype=np.float64)
    dy = np.abs(np.diff(ys))  # length = len(chain) - 1
    cut = len(chain)
    # dy[i+1] must stay in bounds: i+1 <= len(dy)-1 = len(chain)-2
    for i in range(len(chain) - 3, -1, -1):
        if dy[i] > max_jump and dy[i + 1] > max_jump:
            cut = i + 1
            break
    if cut < min_len:
        return chain
    return chain[:cut]


def _embed_merge_masks(masks, reg, emb, min_area, plot_w, plot_h):
    """GOI small-cluster merge (方案 E): re-assign small per-channel mask
    components to the instance whose embedding centroid is nearest.

    masks: list of per-channel plot-local masks (None = dropped channel);
    reg: (K, H, W) plot-local probabilities; emb: (E, H, W) L2-normalized
    full-image embeddings (cropped to the plot extent here).

    Components with area in [8, max(floor, min_area)) are re-assigned to the
    channel with the highest embedding cosine score (> 0.5); otherwise they
    keep their own channel (the min_area drop already ran before this).
    """
    from skimage.measure import label as sk_label

    K = len(masks)
    ph, pw = plot_h, plot_w
    for m in masks:
        if m is not None:
            ph, pw = m.shape
            break
    emb_c = emb[:, :ph, :pw]
    ctr, ok = [], []
    for c in range(K):
        m = masks[c]
        if m is None:
            ctr.append(None); ok.append(False); continue
        conf = (m > 0) & (reg[c] > 0.6)
        if int(conf.sum()) < 4:
            ctr.append(None); ok.append(False); continue
        cv = emb_c[:, conf].mean(axis=1)
        n = float(np.linalg.norm(cv))
        ctr.append(cv / n if n > 1e-6 else None); ok.append(n > 1e-6)
    out = [m.copy() if m is not None else None for m in masks]
    floor = max(8, min_area // 2) if min_area > 0 else 8
    for c in range(K):
        m = masks[c]
        if m is None:
            continue
        lab, n_comp = sk_label(m, connectivity=1, return_num=True)
        if n_comp <= 1:
            continue
        for i in range(1, n_comp + 1):
            comp = lab == i
            area = int(comp.sum())
            if area < 8 or (min_area > 0 and area >= min_area):
                continue
            if area >= floor:
                continue  # not small enough to merge
            px = emb_c[:, comp]
            if px.shape[1] < 2:
                continue
            cv = px.mean(axis=1)
            nv = float(np.linalg.norm(cv))
            if nv < 1e-6:
                continue
            cv = cv / nv
            scores = [float(ctr[j] @ cv) if ok[j] else -2.0 for j in range(K)]
            best = int(np.argmax(scores))
            if scores[best] > 0.5 and best != c:
                out[best][comp] = 1
                out[c][comp] = 0
    return out


def _resolve_approach_zones(masks, reg, emb, dist: int = 4,
                            min_gap: float = 0.10, conf_thr: float = 0.6):
    """P1: approach-zone ownership resolution.

    Independent multi-label masks keep BOTH channels' pixels where two
    curves overlap/touch (measured: 357 px overlap for one val image);
    the skeleton then fuses there and the direction-greedy tracer
    switches curves (17/31 of the >5% failures are crossing_jump, 100%
    of the error lies in approach zones).  For every pixel activated by
    MORE THAN ONE channel we assign it to the channel with the highest
    probability (per-pixel argmax over the active channels), but only
    when the probability gap exceeds ``min_gap`` (otherwise both keep
    the pixel, deferring to the tracer).  Non-overlap pixels are left
    multi-label, preserving the Phase-C independent-mask benefit.

    NOTE: the GOI embedding head of the current checkpoint is NOT
    discriminative enough for this decision (centroid cosine ~0.9996
    between close curves, measured on val img_0069), so the probability
    signal is used instead of embedding cosines.
    """
    K = len(masks)
    out = [m.copy() if m is not None else None for m in masks]
    for c in range(K):
        m = masks[c]
        if m is None:
            continue
        for j in range(c + 1, K):
            mj = masks[j]
            if mj is None:
                continue
            overlap = (m > 0) & (mj > 0)
            if int(overlap.sum()) < 4:
                continue
            diff = reg[c][overlap] - reg[j][overlap]
            idx = np.nonzero(overlap)
            for i in range(len(idx[0])):
                yy, xx = idx[0][i], idx[1][i]
                if diff[i] > min_gap:
                    out[j][yy, xx] = 0
                elif diff[i] < -min_gap:
                    out[c][yy, xx] = 0
    return out


def extract_curves_multi(    image_bgr: np.ndarray,
    structure: ChartStructure,
    x_axis: AxisSpec,
    y_axis: AxisSpec,
    cfg: Optional[Dict] = None,
    segmenter: Optional["object"] = None,
) -> List[Curve]:
    """Extract ALL curve instances (Phase C).

    ``segmenter`` must expose ``prob_full(image) -> (K, H, W)``
    (MultiUNetSegmenter).  Each channel is thresholded, fragment-
    filtered, skeleton-traced and sub-pixel refined exactly like the
    single-curve U-Net path; empty channels are skipped.  Curves are
    returned in channel order (left-to-right at training time).
    """
    cfg = cfg or {}
    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    if plot_w < 20 or plot_h < 20:
        raise CurveExtractionError("plot area too small")
    # B-5a/Phase C tuning: the acceptance threshold 0.5 drops curve
    # portions the model is unsure about (e.g. a steep tail at 0.44),
    # which then misleads the tracer; 0.3 recovers them.  Jump
    # truncation cuts chains that left the curve (sustained |dy| jumps).
    thr = float(cfg.get("multi_mask_thr", 0.3))
    trunc = bool(cfg.get("multi_truncate_jumps", True))
    refine = bool(cfg.get("multi_refine", True))

    prob = segmenter.prob_full(image_bgr)  # (K, H, W)
    if prob.ndim != 3 or prob.shape[0] < 1:
        raise CurveExtractionError("multi segmenter must return (K,H,W)")
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # Phase C: overlapping channels (curves crossing/touching) must not
    # both claim the same pixel -- assign each pixel to the channel with
    # the highest probability (argmax over the plot region), preventing
    # the skeleton tracer from switching curves at crossings.
    reg = prob[:, y0 : y1 + 1, x0 : x1 + 1]  # (K, H, W)
    amax = np.argmax(reg, axis=0)  # winner channel per pixel
    # Candidate C Phase 1 (multi-label): keep every pixel above the
    # threshold in EVERY channel (no argmax exclusion), so crossing
    # regions are not stripped from either curve.  Direction-continuity
    # tracing then resolves the junction.  Default stays argmax for
    # backward compatibility / A-B comparison (PHASE D ablation A5).
    independent = bool(cfg.get("multi_independent_mask", False))
    min_area = int(cfg.get("multi_min_area", 0))
    # GOI embedding merge (方案 E): small components are re-assigned to the
    # instance whose embedding centroid they are nearest to, instead of being
    # dropped outright (ChartZero small-cluster merge).  Requires a checkpoint
    # with an embedding head and multi_embed_merge: true.
    embed_merge = bool(cfg.get("multi_embed_merge", False))
    emb = None
    if embed_merge:
        emb = getattr(segmenter, "embed_full", lambda im: None)(image_bgr)
        if emb is None:
            embed_merge = False  # checkpoint has no embedding head
    # ---- pass 1: build per-channel masks ----
    masks: List[Optional[np.ndarray]] = []
    for c in range(prob.shape[0]):
        region = reg[c]
        if float(region.max()) < thr:
            masks.append(None)
            continue
        if independent:
            mask01 = (region > thr).astype(np.uint8)
        else:
            mask01 = ((region > thr) & (amax == c)).astype(np.uint8)
        mask01 = _filter_mask_fragments(mask01, plot_w, plot_h)
        if int(mask01.sum()) < 16:
            masks.append(None)
            continue
        if min_area > 0 and int(mask01.sum()) < min_area:
            masks.append(None)
            continue
        masks.append(mask01)
    if embed_merge:
        masks = _embed_merge_masks(masks, reg, emb, min_area, plot_w, plot_h)
        if bool(cfg.get("multi_zone_resolve", False)):
            masks = _resolve_approach_zones(masks, reg, emb)
    # ---- pass 2: trace each surviving channel ----
    # P1: approach-zone mask per channel = pixels of this mask within
    # `refine_radius` of ANY other channel's mask; _refine_chain skips the
    # window centroid there (it gets pulled toward the other curve, moving
    # curves out of the 1-2% bucket; recall 0.7352 -> 0.7481 when off).
    refine_radius = int(cfg.get("multi_refine_radius", 6))
    from scipy.ndimage import binary_dilation as _bin_dil
    near_masks = []
    for c in range(len(masks)):
        m = masks[c]
        if m is None:
            near_masks.append(None)
            continue
        near_others = np.zeros(m.shape, dtype=bool)
        for j in range(len(masks)):
            if j != c and masks[j] is not None:
                near_others |= _bin_dil(masks[j].astype(bool), iterations=refine_radius)
        near_masks.append((m > 0) & near_others)
    curves: List[Curve] = []
    for c, mask01 in enumerate(masks):
        if mask01 is None:
            continue
        region = reg[c]
        skel = skeletonize(mask01.astype(bool)).astype(np.uint8)
        # P1b rejected: prob-guided trace (even gated to approach zones)
        # regresses 0.7870 -> 0.7778 -- channel probability is not
        # discriminative for branch exits either; >5% bucket is deferred to
        # training-side fixes (P2 hard-example augmentation).
        chain = _trace_chain(skel)
        if chain is not None:
            xs = [p[0] for p in chain]
            if (max(xs) - min(xs) + 1) < 0.7 * plot_w:
                chain = None
        if chain is not None and refine:
            chain = _refine_chain(region, chain, skip_mask=near_masks[c])
        elif chain is None:
            chain = _column_centroid(region)
        if not chain or len(chain) < 8:
            continue
        if trunc:
            chain = _truncate_jumps(chain)
        # S1b: empirical pixel-bias calibration (measured +0.67 px systematic
        # offset of the model's probability peak on the synthetic set;
        # ~1.34% rel on log axes).  Applied in pixel space before mapping.
        bias_y = float(cfg.get("multi_refine_bias_y", 0.0))
        if bias_y:
            chain = [(px, py + bias_y) for px, py in chain]
        chain = downsample_chain(chain, int(cfg.get("max_points", 2000)))
        points = [(x_axis.pixel_to_value(x0 + px), y_axis.pixel_to_value(y0 + py))
                  for px, py in chain]
        pixel_points = [(x0 + int(round(px)), y0 + int(round(py)))
                        for px, py in chain]
        color_px = rgb[mask01 > 0]
        color = tuple(int(v) for v in np.median(color_px, axis=0)) if len(color_px) else (0, 0, 0)
        curves.append(Curve(name=f"curve_{c}", points=points,
                          pixel_points=pixel_points, color=color,
                          legend_label=None))
    if not curves:
        raise CurveExtractionError("no curve instance found in plot region")
    return curves
