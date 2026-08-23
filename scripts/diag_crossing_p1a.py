"""P1a crossing-failure diagnosis.

Question: are the >5% rel_rmse failures of the GOI multi-curve model caused by
curve crossings / close approaches (曲线交叉/贴近处), i.e. would a crossing
pairing post-processing help?

For every e5p failure (rel > 0.05 or inf) recorded in
``data/eval_multi_diag_goi/diag.json`` (24 images / 31 curves):

  1. run the exact online pipeline on the image:
       detect_structure -> read_ticks(StubOCRBackend) -> build_axes
       -> extract_curves_multi (GOI segmenter)
  2. replicate the pass-1 mask building (multi_mask_thr=0.3,
     multi_independent_mask=true, multi_min_area=450,
     _embed_merge_masks with the GOI embeddings) to obtain the FINAL
     per-channel mask of the matched prediction channel (identical to the
     online code path);
  3. skeletonize that mask, find branch points (8-neighbourhood degree >= 3),
     cluster them into "crossing junctions" (merge branch points within
     3 px), record junction positions;
  4. error attribution in the pixel domain:
       - GT curve: data coords -> pixels via x_axis.value_to_pixel /
         y_axis.value_to_pixel (inverse of pixel_to_value);
       - pred chain: the curve's pixel_points;
       - per pred point: distance to the GT polyline;
       - fraction of |err|>3 px points inside any junction R=8 px
         neighbourhood; SSE contribution of the junction neighbourhood;
  5. classify the failure into a verdict:
       crossing_jump | crossing_merge | partial_trace | systematic | other;
  6. write a visualization PNG (image + channel mask contours + junction
     red dots + pred chain vs GT chain + error points) per failure.

Output:
  data/experiments_p1a/diag_crossing.json  (structured results)
  data/experiments_p1a/vis/*.png           (visualizations)
  data/experiments_p1a/CONCLUSION.md       (written separately)

Usage:
  python scripts/diag_crossing_p1a.py
"""
from __future__ import annotations

import csv as csvlib
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np
from skimage.morphology import skeletonize

from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import (
    _column_centroid,
    _embed_merge_masks,
    _filter_mask_fragments,
    _refine_chain,
    _trace_chain,
    _truncate_jumps,
)
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import downsample_chain, read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(ROOT, "data", "eval_multi_diag_goi", "diag.json")
CKPT = os.path.join(ROOT, "models", "checkpoints", "unet_multi_goi.pt")
OUT_DIR = os.path.join(ROOT, "data", "experiments_p1a")
VIS_DIR = os.path.join(OUT_DIR, "vis")

ERR_PX = 3.0      # pixel error threshold
JUNC_R = 8.0      # junction neighbourhood radius (px)
MERGE_R = 3.0     # branch-point clustering radius (px)
OTHER_PX = 5.0    # pred point within this distance of ANOTHER channel mask = switch zone
GT_APPROACH_PX = 6.0  # GT curve within this distance of another GT curve = crossing/approach

# ---------------------------------------------------------------------------
# GT / helpers
# ---------------------------------------------------------------------------


def load_gt_curves(stem: str) -> list:
    """Same as eval_multi_diag.load_gt_curves: [(label, data-pts), ...]."""
    out = []
    with open(stem + "_curves.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data["curves"]:
        csv_path = os.path.join(os.path.dirname(stem), c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        pts = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        out.append((c.get("label") or c["curve_id"], pts))
    return out


def polyline_dist(points: np.ndarray, poly: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Min distance from each query point to any segment of the polyline."""
    points = np.asarray(points, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 2:
        return np.full(points.shape[0], np.inf)
    seg = poly[1:] - poly[:-1]
    len2 = (seg ** 2).sum(axis=1)
    len2[len2 < 1e-12] = 1e-12
    out = np.full(points.shape[0], np.inf)
    for i in range(0, points.shape[0], chunk):
        q = points[i:i + chunk]
        qa = q[:, None, :] - poly[:-1][None, :, :]
        t = (qa * seg[None, :, :]).sum(axis=2) / len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        proj = poly[:-1][None, :, :] + t[:, :, None] * seg[None, :, :]
        d = np.sqrt(((q[:, None, :] - proj) ** 2).sum(axis=2))
        out[i:i + chunk] = d.min(axis=1)
    return out


# ---------------------------------------------------------------------------
# Junction (crossing-node) detection
# ---------------------------------------------------------------------------


def find_branch_points(mask: np.ndarray) -> list:
    """Skeleton branch points: 8-neighbourhood degree >= 3."""
    skel = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(skel)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if len(pts) < 4:
        return []
    out = []
    for (x, y) in pts:
        deg = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) == (0, 0):
                    continue
                if (x + dx, y + dy) in pts:
                    deg += 1
        if deg >= 3:
            out.append((float(x), float(y)))
    return out


def cluster_junctions(branch_pts: list, merge_r: float = MERGE_R) -> list:
    """Cluster branch points (merge within merge_r px) into junction centers."""
    n = len(branch_pts)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    bp = np.asarray(branch_pts, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.hypot(bp[i, 0] - bp[j, 0], bp[i, 1] - bp[j, 1])) <= merge_r:
                union(i, j)
    clusters: dict = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    return [tuple(np.mean(bp[idx], axis=0).tolist()) for idx in clusters.values()]


# ---------------------------------------------------------------------------
# Pass-1 replication (must mirror extract_curves_multi)
# ---------------------------------------------------------------------------


def build_pass1_masks(prob: np.ndarray, emb, structure, cfg) -> list:
    """Replicate extract_curves_multi pass 1 -> final per-channel masks
    (plot-local), same order/None semantics as the online path."""
    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    thr = float(cfg.get("multi_mask_thr", 0.3))
    independent = bool(cfg.get("multi_independent_mask", False))
    min_area = int(cfg.get("multi_min_area", 0))
    reg = prob[:, y0:y1 + 1, x0:x1 + 1]
    amax = np.argmax(reg, axis=0)
    masks = []
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
    if emb is not None:
        masks = _embed_merge_masks(masks, reg, emb, min_area, plot_w, plot_h)
    return masks


def replicate_curves(prob: np.ndarray, masks: list, structure, x_axis, y_axis, cfg) -> list:
    """Replicate extract_curves_multi pass 2 (trace + refine + truncate +
    downsample + mapping) so the pixel chains exactly match the online
    output without a second model forward pass."""
    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    trunc = bool(cfg.get("multi_truncate_jumps", True))
    reg = prob[:, y0:y1 + 1, x0:x1 + 1]
    curves = []
    for c, mask01 in enumerate(masks):
        if mask01 is None:
            continue
        region = reg[c]
        skel = skeletonize(mask01.astype(bool)).astype(np.uint8)
        chain = _trace_chain(skel)
        if chain is not None:
            xs = [p[0] for p in chain]
            if (max(xs) - min(xs) + 1) < 0.7 * plot_w:
                chain = None
        if chain is not None:
            chain = _refine_chain(region, chain)
        else:
            chain = _column_centroid(region)
        if not chain or len(chain) < 8:
            continue
        if trunc:
            chain = _truncate_jumps(chain)
        chain = downsample_chain(chain, int(cfg.get("max_points", 2000)))
        points = [(x_axis.pixel_to_value(x0 + px), y_axis.pixel_to_value(y0 + py))
                  for px, py in chain]
        pixel_points = [(x0 + int(round(px)), y0 + int(round(py)))
                        for px, py in chain]
        curves.append({"points": points, "pixel_points": pixel_points, "channel": c})
    return curves


# ---------------------------------------------------------------------------
# Error attribution / verdict
# ---------------------------------------------------------------------------


def analyze_curve(pred_px: np.ndarray, gt_px: np.ndarray, mask: np.ndarray,
                  junctions: list, other_masks: list, other_gt_polys: list,
                  x0: float, y0: float) -> dict:
    """Pixel-domain error attribution for one (pred chain, GT curve) pair.

    Evidence families:
      (A) skeleton-junction proximity on the failed channel mask (R=8 px);
      (B) proximity to OTHER channel masks (pred within OTHER_PX of another
          channel's mask = switch zone, incl. close approaches);
      (C) GT-vs-GT approach zones (the failed GT curve passes within
          GT_APPROACH_PX of another GT curve = ground-truth crossing);
      (D) "follows other GT": pred point >3 px from its own GT yet closer to
          another GT polyline by >1 px = the chain is tracing the WRONG curve
          (jump / whole-curve misassignment).
    """
    n = len(pred_px)
    d = polyline_dist(pred_px, gt_px)
    err = d > ERR_PX
    n_err = int(err.sum())
    frac_err = float(n_err / n) if n else 0.0
    sse = float((d ** 2).sum())

    # ---- (A) skeleton-junction proximity ----
    junc_arr = np.asarray(junctions, dtype=np.float64) if junctions else np.zeros((0, 2))
    if len(junc_arr):
        jd = np.sqrt(((pred_px[:, None, :] - junc_arr[None, :, :]) ** 2).sum(axis=2))
        dj = jd.min(axis=1)
    else:
        dj = np.full(n, np.inf)
    in_junc = dj <= JUNC_R
    frac_err_in_junc = float((err & in_junc).sum() / n_err) if n_err else 0.0
    sse_junc = float((d[in_junc] ** 2).sum()) if sse > 0 else 0.0
    junc_sse_ratio = sse_junc / sse if sse > 0 else 0.0

    # ---- (B) other-channel mask proximity (switch zones) ----
    d_other = np.full(n, np.inf)
    for om in other_masks:
        if om is None or om.sum() == 0:
            continue
        dtm = cv2.distanceTransform((om > 0).astype(np.uint8), cv2.DIST_L2, 3)
        ph, pw = om.shape
        px = np.clip((pred_px[:, 0] - x0).astype(int), 0, pw - 1)
        py = np.clip((pred_px[:, 1] - y0).astype(int), 0, ph - 1)
        d_other = np.minimum(d_other, dtm[py, px].astype(np.float64))
    near_other = d_other <= OTHER_PX
    frac_err_near_other = float((err & near_other).sum() / n_err) if n_err else 0.0
    sse_other = float((d[near_other] ** 2).sum()) if sse > 0 else 0.0
    other_sse_ratio = sse_other / sse if sse > 0 else 0.0

    # ---- (C) GT-GT approach zones (ground truth crossings / close passes) ----
    in_gt_app = np.zeros(n, bool)
    gt_app_zones = []
    if len(other_gt_polys) and len(gt_px) >= 2:
        d_g = np.full(len(gt_px), np.inf)
        for og in other_gt_polys:
            if len(og) < 2:
                continue
            d_g = np.minimum(d_g, polyline_dist(gt_px, og))
        approach = d_g <= GT_APPROACH_PX
        if approach.any():
            idx = np.where(approach)[0]
            runs = []
            start = prev = idx[0]
            for k in idx[1:]:
                if k - prev > 3:
                    runs.append((start, prev))
                    start = k
                prev = k
            runs.append((start, prev))
            for (s, e) in runs:
                zc = gt_px[s:e + 1].mean(axis=0)
                gt_app_zones.append([float(zc[0]), float(zc[1])])
            az = gt_px[approach]
            if len(az):
                for i in range(0, n, 512):
                    q = pred_px[i:i + 512]
                    dd = np.sqrt(((q[:, None, :] - az[None, :, :]) ** 2).sum(axis=2))
                    in_gt_app[i:i + 512] = dd.min(axis=1) <= JUNC_R
    frac_err_in_gt_app = float((err & in_gt_app).sum() / n_err) if n_err else 0.0
    sse_gtapp = float((d[in_gt_app] ** 2).sum()) if sse > 0 else 0.0
    gt_app_sse_ratio = sse_gtapp / sse if sse > 0 else 0.0

    # ---- (D) follows another GT curve (jump / misassignment evidence) ----
    # Tight criterion: the pred point must actually lie ON the other GT curve
    # (<=4 px), not merely be "closer to it than to its own GT" (which would
    # flag points that run in empty space far from every curve).
    on_other_gt = np.zeros(n, bool)
    if len(other_gt_polys):
        d_og = np.full(n, np.inf)
        for og in other_gt_polys:
            if len(og) < 2:
                continue
            d_og = np.minimum(d_og, polyline_dist(pred_px, og))
        on_other_gt = (d > ERR_PX) & (d_og <= 4.0)
    frac_err_on_other_gt = float(on_other_gt.sum() / n_err) if n_err else 0.0
    err_mean_err_pts = float(d[err].mean()) if n_err else 0.0

    # ---- unified crossing/approach zone ----
    in_zone = in_junc | near_other | in_gt_app
    frac_err_in_zone = float((err & in_zone).sum() / n_err) if n_err else 0.0
    sse_zone = float((d[in_zone] ** 2).sum()) if sse > 0 else 0.0
    zone_sse_ratio = sse_zone / sse if sse > 0 else 0.0

    # ---- error runs ----
    run_info = []
    if n_err:
        idx = np.where(err)[0]
        start = prev = idx[0]
        for k in idx[1:]:
            if k - prev > 1:
                run_info.append((start, prev))
                start = k
            prev = k
        run_info.append((start, prev))
    runs = []
    for (s, e) in run_info:
        seg = np.arange(s, e + 1)
        runs.append({
            "start": int(s), "end": int(e), "len": int(e - s + 1),
            "near_junction": bool(in_junc[seg].any()),
            "near_other_chan": bool(near_other[seg].any()),
            "near_gt_approach": bool(in_gt_app[seg].any()),
            "follows_other_gt": bool((on_other_gt[seg].sum() / len(seg)) >= 0.5),
            "on_other_gt_frac": round(float(on_other_gt[seg].mean()), 3),
            "mean_err": round(float(d[seg].mean()), 3),
        })
    follows_other_runs = [r for r in runs if r["follows_other_gt"]]

    # ---- chain behavior around the primary junction zone (jump vs merge) ----
    primary_zone = None
    if len(junc_arr):
        zone_idx = np.where(in_junc)[0]
        if len(zone_idx):
            blocks = []
            start = prev = zone_idx[0]
            for k in zone_idx[1:]:
                if k - prev > 8:
                    blocks.append((start, prev))
                    start = k
                prev = k
            blocks.append((start, prev))
            best = max(blocks, key=lambda b: float((d[b[0]:b[1] + 1] ** 2).sum()))
            i0, i1 = best
            pre = d[max(0, i0 - 15):i0]
            mid = d[i0:i1 + 1]
            post = d[i1 + 1:i1 + 16]
            primary_zone = {
                "i0": int(i0), "i1": int(i1),
                "pre_err": float(pre.mean()) if len(pre) else None,
                "mid_err": float(mid.mean()) if len(mid) else None,
                "post_err": float(post.mean()) if len(post) else None,
                "local_sse": float((d[i0:i1 + 1] ** 2).sum()),
                "n_pts": int(i1 - i0 + 1),
            }

    # ---- verdict ----
    # A "follows other GT" run only counts as jump evidence when it is long
    # enough to be a real branch switch (1-4 px runs are noise).
    def _pre_correct(r) -> bool:
        s = r["start"]
        if s >= 15:
            return float(d[s - 15:s].mean()) <= ERR_PX
        if s > 0:
            return float(d[:s].mean()) <= ERR_PX
        return False  # error starts at the very beginning of the chain

    verdict = "other"
    jump_runs = [r for r in runs if r["follows_other_gt"] and r["len"] >= 5]
    if jump_runs:
        # chain follows another GT curve: jump if it was correct before the
        # switch, systematic (channel misassignment) if wrong from the start
        r0 = jump_runs[0]
        if _pre_correct(r0):
            verdict = "crossing_jump"
        else:
            verdict = "systematic"
    elif (frac_err_in_zone >= 0.5 or zone_sse_ratio >= 0.5) \
            and frac_err_on_other_gt < 0.4 and err_mean_err_pts <= 25.0:
        # no wholesale jump; errors concentrated at crossing/approach zones,
        # chain does NOT trace another curve, and deviations are moderate ->
        # the chain cuts through the crossing blob but returns to GT (merge)
        verdict = "crossing_merge"
    elif frac_err >= 0.5:
        verdict = "systematic"
    else:
        verdict = "partial_trace"

    return {
        "n": n,
        "n_err": n_err,
        "frac_err": round(frac_err, 4),
        # junction (A)
        "n_junctions": len(junctions),
        "frac_err_in_junc": round(frac_err_in_junc, 4),
        "junc_sse_ratio": round(junc_sse_ratio, 4),
        # other-channel proximity (B)
        "frac_err_near_other": round(frac_err_near_other, 4),
        "other_sse_ratio": round(other_sse_ratio, 4),
        # GT approach (C)
        "n_gt_approach_zones": len(gt_app_zones),
        "gt_approach_zones": gt_app_zones,
        "frac_err_in_gt_app": round(frac_err_in_gt_app, 4),
        "gt_app_sse_ratio": round(gt_app_sse_ratio, 4),
        # follows-other (D)
        "frac_err_on_other_gt": round(frac_err_on_other_gt, 4),
        # unified
        "frac_err_in_zone": round(frac_err_in_zone, 4),
        "zone_sse_ratio": round(zone_sse_ratio, 4),
        "n_err_runs": len(runs),
        "n_runs_follow_other": len(follows_other_runs),
        "err_runs": runs,
        "mean_px_err": round(float(d.mean()), 4),
        "rmse_px": round(float(np.sqrt(np.mean(d ** 2))), 4),
        "primary_zone": primary_zone,
        "verdict": verdict,
    }


def curve_rmse_value(gt: np.ndarray, pred: np.ndarray) -> float:
    """Same metric as eval_multi_diag.curve_rmse (data space)."""
    g = gt[np.argsort(gt[:, 0])]
    p = pred[np.argsort(pred[:, 0])]
    x_lo = max(float(g[0, 0]), float(p[0, 0]))
    x_hi = min(float(g[-1, 0]), float(p[-1, 0]))
    inside = (p[:, 0] >= x_lo) & (p[:, 0] <= x_hi)
    if inside.sum() < 2:
        return float("inf")
    gy = np.interp(p[inside, 0], g[:, 0], g[:, 1])
    return float(np.sqrt(np.mean((p[inside, 1] - gy) ** 2)))


def deep_attribution(image_gray: np.ndarray, pred_px: np.ndarray,
                     gt_px: np.ndarray, other_gt_polys: list,
                     junctions: list, err_runs: list) -> dict:
    """Extra diagnostic quantities that decide whether an error is really
    caused by a GT-vs-GT crossing / close approach:

      gt_ink_frac      - GT pixels within 3 px of image ink (sanity of the
                         value_to_pixel conversion; should be ~1).
      gt_pair_min_dist - min distance between the failed GT curve and any
                         other GT curve (0 -> true crossing/touch exists).
      gt_approach_len  - length (px along GT) of the closest GT-GT approach.
      followed_gt      - which other GT curve the error points are nearest to
                         (the curve the pred chain is tracing instead).
      switch           - geometry of the first 'follows other GT' error run:
                         switch point position, distance to its own GT, to the
                         followed GT, to the nearest skeleton junction and to
                         the nearest GT-GT approach location.
    """
    n = len(pred_px)
    d = polyline_dist(pred_px, gt_px)
    err = d > ERR_PX

    # ---- GT ink sanity ----
    ink = (image_gray < 128).astype(np.uint8)
    dt_ink = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    gy = np.clip(np.round(gt_px[:, 1]).astype(int), 0, image_gray.shape[0] - 1)
    gx = np.clip(np.round(gt_px[:, 0]).astype(int), 0, image_gray.shape[1] - 1)
    gt_ink_frac = float((dt_ink[gy, gx] <= 3.0).mean())

    # ---- GT-GT approach geometry ----
    gt_pair_min_dist = float("inf")
    gt_approach_len = 0
    approach_pts = []
    for og in other_gt_polys:
        if len(og) < 2:
            continue
        dg = polyline_dist(gt_px, og)
        gt_pair_min_dist = min(gt_pair_min_dist, float(dg.min()))
        near = dg <= GT_APPROACH_PX
        if near.any():
            gt_approach_len += int(near.sum())
            approach_pts.append(gt_px[near])
    if approach_pts:
        approach_pts = np.concatenate(approach_pts, axis=0)
    else:
        approach_pts = np.zeros((0, 2))

    # ---- followed GT (per error point) ----
    followed = {"label_idx": None, "median_d": None, "frac_on": None, "n": 0}
    if err.any() and other_gt_polys:
        best = None
        for gi, og in enumerate(other_gt_polys):
            if len(og) < 2:
                continue
            dg = polyline_dist(pred_px[err], og)
            frac_on = float((dg <= 4.0).mean())
            med = float(np.median(dg))
            if best is None or frac_on > best[1]:
                best = (gi, frac_on, med)
        if best is not None:
            followed = {"label_idx": best[0], "frac_on": round(best[1], 3),
                        "median_d": round(best[2], 3), "n": int(err.sum())}

    # ---- switch-point geometry (first follows-other run) ----
    switch = None
    for r in err_runs:
        if r.get("follows_other_gt"):
            s = r["start"]
            pos = pred_px[s]
            d_own = float(d[s])
            d_oth = min((float(polyline_dist(pos[None, :], og)[0])
                         for og in other_gt_polys if len(og) >= 2),
                        default=float("inf"))
            d_junc = min((float(np.hypot(pos[0] - jp[0], pos[1] - jp[1]))
                          for jp in junctions), default=float("inf"))
            if len(approach_pts):
                d_app = float(np.sqrt(((approach_pts - pos[None, :]) ** 2).sum(axis=1)).min())
            else:
                d_app = float("inf")
            switch = {
                "idx": int(s),
                "pos": [float(pos[0]), float(pos[1])],
                "d_own_gt": round(d_own, 2),
                "d_followed_gt": round(d_oth, 2) if np.isfinite(d_oth) else None,
                "d_nearest_junction": round(d_junc, 2) if np.isfinite(d_junc) else None,
                "d_nearest_gt_approach": round(d_app, 2) if np.isfinite(d_app) else None,
                "run_len": r["len"],
            }
            break
    return {
        "gt_ink_frac": round(gt_ink_frac, 4),
        "gt_pair_min_dist": round(gt_pair_min_dist, 3),
        "gt_approach_len_px": int(gt_approach_len),
        "followed_gt": followed,
        "switch": switch,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_vis(image_rgb: np.ndarray, structure, masks: list, channels: list,
             junctions: list, pred_px: np.ndarray, gt_px: np.ndarray,
             d: np.ndarray, title: str, path: str,
             gt_app_zones: list = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_app_zones = gt_app_zones or []

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(16, 8))
    ax.imshow(image_rgb)
    # per-channel mask contours
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]
    for k, ch in enumerate(channels):
        m = masks[ch]
        if m is None:
            continue
        x0, y0, x1, y1 = structure.plot_bbox
        full = np.zeros(image_rgb.shape[:2], np.uint8)
        full[y0:y1 + 1, x0:x1 + 1] = m
        cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(full, cnts, -1, 1, 1)
        yy, xx = np.nonzero(full)
        if len(xx):
            ax.scatter(xx[::2], yy[::2], s=0.5, c=palette[k % len(palette)], linewidths=0)
    # junctions
    for jp in junctions:
        circ = plt.Circle(jp, JUNC_R, color="red", fill=False, ls="--", lw=1.2)
        ax.add_patch(circ)
        ax.plot(jp[0], jp[1], "ro", ms=7, mec="white", mew=0.5)
    # GT-GT approach zones (magenta)
    for zp in gt_app_zones:
        circ = plt.Circle(zp, JUNC_R, color="magenta", fill=False, ls=":", lw=1.4)
        ax.add_patch(circ)
        ax.plot(zp[0], zp[1], "m^", ms=7, mec="white", mew=0.5)
    # GT chain (green) then pred chain (blue)
    ax.plot(gt_px[:, 0], gt_px[:, 1], "-", color="lime", lw=2.2, alpha=0.9, label="GT")
    ax.plot(pred_px[:, 0], pred_px[:, 1], "-", color="blue", lw=1.8, alpha=0.9, label="pred")
    err = d > ERR_PX
    if err.any():
        ax.scatter(pred_px[err, 0], pred_px[err, 1], s=8, c="yellow", marker="o",
                   edgecolors="black", linewidths=0.3, label=f"err>{ERR_PX}px")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, image_rgb.shape[1])
    ax.set_ylim(image_rgb.shape[0], 0)

    # zoom around the primary junction / error region
    zoom_center = None
    if junctions:
        jp = junctions[0]
        zoom_center = jp
    elif err.any():
        idx = int(np.argmax(d))
        zoom_center = (float(pred_px[idx, 0]), float(pred_px[idx, 1]))
    if zoom_center is not None:
        r = 55
        cx, cy = zoom_center
        x0z, x1z = max(0, int(cx - r)), min(image_rgb.shape[1], int(cx + r))
        y0z, y1z = max(0, int(cy - r)), min(image_rgb.shape[0], int(cy + r))
        axz.imshow(image_rgb[y0z:y1z, x0z:x1z])
        off = np.array([x0z, y0z])
        for k, ch in enumerate(channels):
            m = masks[ch]
            if m is None:
                continue
            x0, y0, x1, y1 = structure.plot_bbox
            full = np.zeros(image_rgb.shape[:2], np.uint8)
            full[y0:y1 + 1, x0:x1 + 1] = m
            cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(full, cnts, -1, 1, 1)
            yy, xx = np.nonzero(full)
            if len(xx):
                axz.scatter(xx[::2] - x0z, yy[::2] - y0z, s=1, c=palette[k % len(palette)], linewidths=0)
        for jp in junctions:
            if x0z <= jp[0] <= x1z and y0z <= jp[1] <= y1z:
                circ = plt.Circle((jp[0] - x0z, jp[1] - y0z), JUNC_R, color="red", fill=False, ls="--", lw=1.2)
                axz.add_patch(circ)
                axz.plot(jp[0] - x0z, jp[1] - y0z, "ro", ms=6, mec="white", mew=0.5)
        g = gt_px - off
        g = g[(g[:, 0] >= 0) & (g[:, 0] < x1z - x0z) & (g[:, 1] >= 0) & (g[:, 1] < y1z - y0z)]
        p = pred_px - off
        p = p[(p[:, 0] >= 0) & (p[:, 0] < x1z - x0z) & (p[:, 1] >= 0) & (p[:, 1] < y1z - y0z)]
        if len(g):
            axz.plot(g[:, 0], g[:, 1], "-", color="lime", lw=2.5)
        if len(p):
            axz.plot(p[:, 0], p[:, 1], "-", color="blue", lw=2.0)
            e = d > ERR_PX
            pe = pred_px[e] - off
            pe = pe[(pe[:, 0] >= 0) & (pe[:, 0] < x1z - x0z) & (pe[:, 1] >= 0) & (pe[:, 1] < y1z - y0z)]
            if len(pe):
                axz.scatter(pe[:, 0], pe[:, 1], s=10, c="yellow", marker="o",
                            edgecolors="black", linewidths=0.3)
        axz.set_title("zoom (junction / max-err)", fontsize=10)
        axz.set_xlim(0, x1z - x0z)
        axz.set_ylim(y1z - y0z, 0)
    else:
        axz.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    with open(DIAG, encoding="utf-8") as f:
        diag = json.load(f)

    # collect e5p failures
    failures = []
    for r in diag["rows"]:
        for g in r.get("per_gt", []):
            rel = g["rel"]
            if rel > 0.05 or rel == float("inf"):
                failures.append({
                    "image": r["image"], "template": r.get("template", "?"),
                    "shape": r.get("shape", "?"), "deg": r.get("deg", ""),
                    "label": g["label"], "pred_chan": g["pred_chan"],
                    "rel": rel, "y_span": g.get("y_span", None),
                })
    print(f"[p1a] e5p failures from diag.json: {len(failures)} "
          f"across {len({f['image'] for f in failures})} images")

    # locate each failure image (val_multi first, then val_single)
    data_dirs = [os.path.join(ROOT, "data", "val_multi"),
                 os.path.join(ROOT, "data", "val_single")]
    img_path_of = {}
    for f in failures:
        name = f["image"]
        if name in img_path_of:
            continue
        for dd in data_dirs:
            p = os.path.join(dd, name)
            if os.path.exists(p):
                img_path_of[name] = p
                break

    os.makedirs(VIS_DIR, exist_ok=True)

    cfg = load_config()
    print(f"[p1a] checkpoint: {CKPT}")
    segmenter = MultiUNetSegmenter(CKPT, size=int(cfg.get("unet_size", 256)))
    cfg["multi_unet_checkpoint"] = CKPT

    # group failures by image
    by_img = defaultdict(list)
    for f in failures:
        by_img[f["image"]].append(f)

    results = []
    for name in sorted(by_img):
        img_path = img_path_of.get(name)
        if img_path is None:
            print(f"[WARN] image not found: {name}")
            continue
        stem = os.path.splitext(img_path)[0]
        flist = by_img[name]
        rec = {"image": name, "template": flist[0]["template"],
               "shape": flist[0]["shape"], "curve_failures": []}
        try:
            img = read_image(img_path)
            structure = detect_structure(img, cfg)
            ocr = StubOCRBackend(stem + "_labels.json")
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            x_axis, y_axis = build_axes(
                x_ticks, y_ticks,
                x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
                y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
            )
            prob = segmenter.prob_full(img)
            emb = segmenter.embed_full(img) if cfg.get("multi_embed_merge") else None
            masks = build_pass1_masks(prob, emb, structure, cfg)
            curves = replicate_curves(prob, masks, structure, x_axis, y_axis, cfg)
            gts = load_gt_curves(stem)

            # map curve index -> raw channel
            surv = [c for c, m in enumerate(masks) if m is not None]
            if len(curves) != len(surv):
                raise RuntimeError(f"curve/channel count mismatch: {len(curves)} vs {len(surv)}")

            # per-channel junction detection (only for surviving channels)
            # NOTE: masks are plot-LOCAL; junction positions must be shifted
            # into full-image coordinates to match pred_px / gt_px.
            x0, y0, x1, y1 = structure.plot_bbox
            junctions_of = {}
            joint_junctions_of = {}
            for c in surv:
                jl = cluster_junctions(find_branch_points(masks[c]))
                junctions_of[c] = [(float(a + x0), float(b + y0)) for a, b in jl]
                # joint skeleton: (failed channel | each other channel) forms a
                # true X at crossings -> degree>=3 nodes appear there even when
                # the single-channel mask only shows a bulge. This is the
                # trigger a pairing post-processing would actually use.
                joint = []
                for o in surv:
                    if o == c or masks[o] is None:
                        continue
                    jj = cluster_junctions(find_branch_points(masks[c] | masks[o]))
                    joint.extend(jj)
                joint_junctions_of[c] = [(float(a + x0), float(b + y0)) for a, b in joint]

            # GT index -> label lookup
            gt_by_label = {lab: pts for lab, pts in gts}

            for f in flist:
                sub = dict(f)
                try:
                    j = f["pred_chan"]
                    if j is None or j >= len(curves):
                        sub.update({"error": f"pred_chan {j} out of range"})
                        rec["curve_failures"].append(sub)
                        continue
                    ch = surv[j]
                    gt_pts = gt_by_label.get(f["label"])
                    if gt_pts is None or len(gt_pts) < 2:
                        sub.update({"error": "GT curve not found / too short"})
                        rec["curve_failures"].append(sub)
                        continue
                    gt_px = np.column_stack([
                        x_axis.value_to_pixel(gt_pts[:, 0]),
                        y_axis.value_to_pixel(gt_pts[:, 1]),
                    ]).astype(np.float64)
                    pred_px = np.asarray(curves[j]["pixel_points"], dtype=np.float64)
                    if len(pred_px) < 2:
                        sub.update({"error": "pred chain too short"})
                        rec["curve_failures"].append(sub)
                        continue
                    junctions = junctions_of[ch]
                    joint_junctions = joint_junctions_of[ch]
                    # other surviving channel masks (plot-local) for switch zones
                    other_masks = [masks[c] for c in surv if c != ch]
                    # other GT curves in pixel space
                    other_gt_polys = [
                        np.column_stack([
                            x_axis.value_to_pixel(pts[:, 0]),
                            y_axis.value_to_pixel(pts[:, 1]),
                        ]).astype(np.float64)
                        for lab, pts in gts if lab != f["label"] and len(pts) >= 2
                    ]
                    ana = analyze_curve(pred_px, gt_px, masks[ch], junctions,
                                        other_masks, other_gt_polys, x0, y0)
                    d = polyline_dist(pred_px, gt_px)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    deep = deep_attribution(img_gray, pred_px, gt_px,
                                            other_gt_polys, junctions,
                                            ana["err_runs"])
                    # how close is the switch point to a JOINT (two-channel)
                    # skeleton junction?  -> viability of the pairing trigger
                    if deep["switch"] is not None:
                        sp = np.asarray(deep["switch"]["pos"])
                        if joint_junctions:
                            dj = min(float(np.hypot(sp[0] - a, sp[1] - b))
                                     for a, b in joint_junctions)
                        else:
                            dj = float("inf")
                        deep["switch"]["d_nearest_joint_junction"] = (
                            round(dj, 2) if np.isfinite(dj) else None)
                    # is the chain a GOOD curve for another GT (matching artifact)?
                    pred_data = np.asarray(curves[j]["points"], dtype=np.float64)
                    best_gt = {"label": None, "rel": None, "rmse": None}
                    for lab2, pts2 in gts:
                        r = curve_rmse_value(pts2, pred_data)
                        if r == float("inf"):
                            continue
                        ysp = float(pts2[:, 1].max() - pts2[:, 1].min())
                        rel2 = r / ysp if ysp > 0 else float("inf")
                        if best_gt["rel"] is None or rel2 < best_gt["rel"]:
                            best_gt = {"label": lab2, "rel": round(rel2, 4),
                                       "rmse": round(r, 4)}

                    safe = "".join(ch_ for ch_ in f["label"] if ch_.isalnum() or ch_ in "_- ") or "curve"
                    vis_name = f"{os.path.splitext(name)[0]}_{safe.strip().replace(' ', '_')}.png"
                    vis_path = os.path.join(VIS_DIR, vis_name)
                    title = (f"{name} | {f['label']} | ch={j} | rel={f['rel']:.3f} | "
                             f"{ana['verdict']} | junc={len(junctions)}")
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    save_vis(image_rgb, structure, masks, surv, junctions,
                             pred_px, gt_px, d, title, vis_path,
                             gt_app_zones=ana["gt_approach_zones"])

                    sub.update({
                        "pred_chan": j,
                        "channel": ch,
                        "rel": f["rel"],
                        "n_junctions": len(junctions),
                        "n_joint_junctions": len(joint_junctions),
                        "junction_positions": [[float(a), float(b)] for a, b in junctions],
                        "err_px_frac_in_junction": ana["frac_err_in_junc"],
                        "junc_sse_ratio": ana["junc_sse_ratio"],
                        "frac_err_points": ana["frac_err"],
                        "frac_err_near_other": ana["frac_err_near_other"],
                        "other_sse_ratio": ana["other_sse_ratio"],
                        "frac_err_in_gt_app": ana["frac_err_in_gt_app"],
                        "gt_app_sse_ratio": ana["gt_app_sse_ratio"],
                        "frac_err_on_other_gt": ana["frac_err_on_other_gt"],
                        "frac_err_in_zone": ana["frac_err_in_zone"],
                        "zone_sse_ratio": ana["zone_sse_ratio"],
                        "n_gt_approach_zones": ana["n_gt_approach_zones"],
                        "gt_approach_zones": ana["gt_approach_zones"],
                        "n_err_runs": ana["n_err_runs"],
                        "n_runs_follow_other": ana["n_runs_follow_other"],
                        "err_runs": ana["err_runs"],
                        "rmse_px": ana["rmse_px"],
                        "mean_px_err": ana["mean_px_err"],
                        "primary_zone": ana["primary_zone"],
                        "verdict": ana["verdict"],
                        "deep": deep,
                        "best_gt": best_gt,
                        "vis": os.path.relpath(vis_path, ROOT),
                        "error": None,
                    })
                    sw = deep["switch"] or {}
                    print(f"[OK ] {name} {f['label']}: ch={j} rel={f['rel']:.4f} "
                          f"junc={len(junctions)} onOther={ana['frac_err_on_other_gt']:.2f} "
                          f"zone={ana['frac_err_in_zone']:.2f} "
                          f"minGtPair={deep['gt_pair_min_dist']:.1f}px "
                          f"swJunc={sw.get('d_nearest_junction')} "
                          f"swApp={sw.get('d_nearest_gt_approach')} -> {ana['verdict']}")
                except Exception as e:
                    sub.update({"error": f"{type(e).__name__}: {e}"})
                    print(f"[ERR] {name} {f['label']}: {sub['error']}")
                rec["curve_failures"].append(sub)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[IMG-ERR] {name}: {rec['error']}")
        results.append(rec)

    # ---- summary ----
    ok_fails = [sf for r in results for sf in r["curve_failures"]
                if sf.get("error") is None]
    verdicts = Counter(sf["verdict"] for sf in ok_fails)
    crossing = verdicts["crossing_jump"] + verdicts["crossing_merge"]
    fracs = [sf["err_px_frac_in_junction"] for sf in ok_fails]
    sse_ratios = [sf["junc_sse_ratio"] for sf in ok_fails]
    zone_fracs = [sf["frac_err_in_zone"] for sf in ok_fails]
    zone_sses = [sf["zone_sse_ratio"] for sf in ok_fails]
    on_other = [sf["frac_err_on_other_gt"] for sf in ok_fails]
    gtapp = [sf["frac_err_in_gt_app"] for sf in ok_fails]

    print("\n================ P1a summary ================")
    print(f"failures analysed: {len(ok_fails)} / {len(failures)}")
    print(f"verdicts: {dict(verdicts)}")
    print(f"crossing-attributed: {crossing}/{len(ok_fails)} = "
          f"{crossing / len(ok_fails):.2%}" if ok_fails else "n/a")
    if fracs:
        print(f"err_px_frac_in_junction (A): mean={np.mean(fracs):.3f} "
              f"median={np.median(fracs):.3f}")
        print(f"junc_sse_ratio (A): mean={np.mean(sse_ratios):.3f} "
              f"median={np.median(sse_ratios):.3f}")
        print(f"frac_err_in_zone (A|B|C): mean={np.mean(zone_fracs):.3f} "
              f"median={np.median(zone_fracs):.3f}")
        print(f"zone_sse_ratio (A|B|C): mean={np.mean(zone_sses):.3f} "
              f"median={np.median(zone_sses):.3f}")
        print(f"frac_err_on_other_gt (D): mean={np.mean(on_other):.3f} "
              f"median={np.median(on_other):.3f}")
        print(f"frac_err_in_gt_app (C): mean={np.mean(gtapp):.3f} "
              f"median={np.median(gtapp):.3f}")
        print(f"zone>=0.5 (A|B|C): {sum(f >= 0.5 for f in zone_fracs)}/{len(zone_fracs)}")
        print(f"on_other>=0.4 (D): {sum(f >= 0.4 for f in on_other)}/{len(on_other)}")
    # template / shape breakdown
    tpl = defaultdict(Counter)
    shp = defaultdict(Counter)
    for sf in ok_fails:
        tpl[sf["verdict"]][sf["template"]] += 1
        shp[sf["verdict"]][sf["shape"]] += 1
    for v in sorted(verdicts):
        print(f"  {v}: templates={dict(tpl[v])} shapes={dict(shp[v])}")

    out = {
        "meta": {
            "script": "scripts/diag_crossing_p1a.py",
            "diag_source": DIAG,
            "checkpoint": CKPT,
            "config": {k: cfg.get(k) for k in
                       ("multi_mask_thr", "multi_independent_mask",
                        "multi_min_area", "multi_embed_merge", "multi_truncate_jumps")},
            "err_px_threshold": ERR_PX,
            "junction_radius_px": JUNC_R,
            "merge_radius_px": MERGE_R,
            "other_chan_proximity_px": OTHER_PX,
            "gt_approach_px": GT_APPROACH_PX,
            "gt_pixel_conversion": "x_axis.value_to_pixel / y_axis.value_to_pixel",
        },
        "summary": {
            "n_failures": len(failures),
            "n_images": len(by_img),
            "n_analysed": len(ok_fails),
            "verdicts": dict(verdicts),
            "n_crossing": crossing,
            "crossing_fraction": round(crossing / len(ok_fails), 4) if ok_fails else None,
            "err_frac_in_junc_mean": round(float(np.mean(fracs)), 4) if fracs else None,
            "err_frac_in_junc_median": round(float(np.median(fracs)), 4) if fracs else None,
            "junc_sse_ratio_mean": round(float(np.mean(sse_ratios)), 4) if sse_ratios else None,
            "zone_frac_mean": round(float(np.mean(zone_fracs)), 4) if zone_fracs else None,
            "zone_sse_ratio_mean": round(float(np.mean(zone_sses)), 4) if zone_sses else None,
            "on_other_gt_mean": round(float(np.mean(on_other)), 4) if on_other else None,
            "n_zone_ge_0p5": sum(f >= 0.5 for f in zone_fracs),
            "n_on_other_ge_0p4": sum(f >= 0.4 for f in on_other),
            "verdict_by_template": {v: dict(c) for v, c in tpl.items()},
            "verdict_by_shape": {v: dict(c) for v, c in shp.items()},
        },
        "images": results,
    }
    out_path = os.path.join(OUT_DIR, "diag_crossing.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n[p1a] wrote {out_path}")
    print(f"[p1a] wrote {len(ok_fails)} visualizations to {VIS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
