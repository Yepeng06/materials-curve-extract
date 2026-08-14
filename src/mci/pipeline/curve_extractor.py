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
def _trace_chain(skel: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """Walk the skeleton from the leftmost endpoint with direction continuity.

    At junctions (grid crossings) the neighbour that continues the current
    direction is preferred, so the walk follows the curve instead of turning
    onto a crossing line.
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

            def score(n):
                dx, dy = n[0] - cur[0], n[1] - cur[1]
                return (-(dx * dir0[0] + dy * dir0[1]), -n[0], n[1])
        else:
            def score(n):
                return (-n[0], n[1])
        cand.sort(key=score)
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
                  radius: int = 6) -> List[Tuple[float, float]]:
    """Sub-pixel refinement: local probability-weighted centroid around each
    skeleton point.  A small window keeps the centroid on the curve even on
    steep segments (unlike per-column centroids)."""
    h, w = prob.shape
    out: List[Tuple[float, float]] = []
    for x, y in chain:
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
def _select_curve_mask_cv(image_bgr: np.ndarray, structure: ChartStructure,
                          cfg: Dict) -> np.ndarray:
    """CV path: binarize + grid removal + dash bridging + component scoring."""
    from .segmenter import cv_plot_mask

    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    ink = cv_plot_mask(image_bgr, structure, cfg)

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
