"""Chain-model pixel-level failure diagnosis (Phase C diagnostic, 2026).

Question: for the chain multi-curve model (unet_multi_chain.pt, val recall
0.85, >5% bucket 18 curves, 2-5% bucket 34), is the dominant failure
mechanism still the GOI-era "close-approach zone attribution error" (61% of
e5p), or has it shifted (whole-chain offset on low-quality screenshots,
marker interference, steep tails, axis mapping, ghost channels)?

For every e5p and e2-5 failure curve recorded in
``data/eval_multi_diag_chain_px/diag.json``:

  1. run the exact online pipeline on the image:
       detect_structure -> read_ticks(StubOCRBackend) -> build_axes
       -> replicate extract_curves_multi pass 1 + pass 2 EXACTLY as the
       online path (multi_independent_mask, multi_min_area=450,
       multi_embed_merge, P1 skip-mask refine radius=6, bias_y=-0.67,
       multi_truncate_jumps, downsample);
  2. GT pixel chains in TWO ways:
       gt_px_pipe : pipeline value_to_pixel (what the eval metric "sees");
       gt_px_meta : _meta.json curves_px (authoritative generator pixels,
                    used to detect axis-mapping errors via axis_dev);
  3. pixel error attribution for the matched pred chain (same evidence
     families as p1a: skeleton junctions, other-channel proximity, GT-GT
     approach zones, on-other-GT fraction, error runs) plus an offset
     profile (uniform offset vs slope/scale error);
  4. classify: crossing_jump | crossing_merge | systematic | spike |
     missing | ghost (with axis_error flag when the pipeline GT mapping
     deviates from the true ink and the pred matches the ink);
  5. write visualizations for e5p curves.

Output:
  data/experiments_chain_px/diag_chain_px.json
  data/experiments_chain_px/vis/*.png

Usage:
  python scripts/diag_chain_px.py
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
DIAG = os.path.join(ROOT, "data", "eval_multi_diag_chain_px", "diag.json")
CKPT = os.path.join(ROOT, "models", "checkpoints", "unet_multi_chain.pt")
OUT_DIR = os.path.join(ROOT, "data", "experiments_chain_px")
VIS_DIR = os.path.join(OUT_DIR, "vis")

ERR_PX = 3.0          # pixel error threshold
JUNC_R = 8.0          # junction neighbourhood radius (px)
MERGE_R = 3.0         # branch-point clustering radius (px)
OTHER_PX = 5.0        # pred point within this distance of ANOTHER channel mask
GT_APPROACH_PX = 6.0  # GT curve within this distance of another GT curve
AXIS_DEV_PX = 2.5     # pipeline GT vs meta GT mean deviation -> axis suspicion
IMG_DIRS = ["val_multi", "val_single"]


# ---------------------------------------------------------------------------
# GT / helpers
# ---------------------------------------------------------------------------


def load_gt_curves(stem: str) -> list:
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


def load_meta_curves_px(stem: str) -> list:
    """GT pixel chains straight from the generator (_meta.json curves_px),
    in the same order as _curves.json curves."""
    with open(stem + "_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return [np.asarray(c, dtype=np.float64) for c in meta.get("curves_px", [])]


def polyline_dist(points: np.ndarray, poly: np.ndarray, chunk: int = 512) -> np.ndarray:
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


def find_branch_points(mask: np.ndarray) -> list:
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
# Online replication (must mirror src/mci/pipeline/curve_extractor.py
# extract_curves_multi lines 745-855 for the chain config)
# ---------------------------------------------------------------------------


def build_pass1_masks(prob: np.ndarray, emb, structure, cfg) -> list:
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
    """Pass-2 replication with the P1 skip-mask refine and the S1b bias_y."""
    x0, y0, x1, y1 = structure.plot_bbox
    plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1
    trunc = bool(cfg.get("multi_truncate_jumps", True))
    refine = bool(cfg.get("multi_refine", True))
    refine_radius = int(cfg.get("multi_refine_radius", 6))
    bias_y = float(cfg.get("multi_refine_bias_y", 0.0))
    reg = prob[:, y0:y1 + 1, x0:x1 + 1]

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
        if chain is not None and refine:
            chain = _refine_chain(region, chain, skip_mask=near_masks[c])
        elif chain is None:
            chain = _column_centroid(region)
        if not chain or len(chain) < 8:
            continue
        if trunc:
            chain = _truncate_jumps(chain)
        if bias_y:
            chain = [(px, py + bias_y) for px, py in chain]
        chain = downsample_chain(chain, int(cfg.get("max_points", 2000)))
        points = [(x_axis.pixel_to_value(x0 + px), y_axis.pixel_to_value(y0 + py))
                  for px, py in chain]
        pixel_points = [(x0 + int(round(px)), y0 + int(round(py)))
                        for px, py in chain]
        curves.append({"points": points, "pixel_points": pixel_points, "channel": c})
    return curves


# ---------------------------------------------------------------------------
# Pixel error attribution
# ---------------------------------------------------------------------------


def analyze_curve(pred_px: np.ndarray, gt_px: np.ndarray, gt_px_meta: np.ndarray,
                  mask: np.ndarray, junctions: list, other_masks: list,
                  other_gt_polys: list, x0: float, y0: float) -> dict:
    n = len(pred_px)
    d = polyline_dist(pred_px, gt_px)
    err = d > ERR_PX
    n_err = int(err.sum())
    frac_err = float(n_err / n) if n else 0.0
    sse = float((d ** 2).sum())

    # meta-GT (true ink) distance for axis-error detection
    d_meta = polyline_dist(pred_px, gt_px_meta) if len(gt_px_meta) >= 2 else d
    err_meta = d_meta > ERR_PX
    frac_err_meta = float(err_meta.sum() / n) if n else 0.0
    # pipeline GT vs true ink deviation
    if len(gt_px) >= 2 and len(gt_px_meta) >= 2:
        axis_dev = float(polyline_dist(gt_px, gt_px_meta).mean())
    else:
        axis_dev = 0.0

    # ---- (A) skeleton-junction proximity ----
    junc_arr = np.asarray(junctions, dtype=np.float64) if junctions else np.zeros((0, 2))
    if len(junc_arr):
        jd = np.sqrt(((pred_px[:, None, :] - junc_arr[None, :, :]) ** 2).sum(axis=2))
        dj = jd.min(axis=1)
    else:
        dj = np.full(n, np.inf)
    in_junc = dj <= JUNC_R
    frac_err_in_junc = float((err & in_junc).sum() / n_err) if n_err else 0.0
    junc_sse_ratio = float((d[in_junc] ** 2).sum() / sse) if sse > 0 else 0.0

    # ---- (B) other-channel mask proximity ----
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
    other_sse_ratio = float((d[near_other] ** 2).sum() / sse) if sse > 0 else 0.0

    # ---- (C) GT-GT approach zones (on the TRUE meta GT, so the crossing
    # geometry is generator-truth, independent of the pipeline mapping) ----
    in_gt_app = np.zeros(n, bool)
    gt_app_zones = []
    gt_pair_min_dist = float("inf")
    gt_approach_len = 0
    if len(other_gt_polys) and len(gt_px_meta) >= 2:
        d_g = np.full(len(gt_px_meta), np.inf)
        for og in other_gt_polys:
            if len(og) < 2:
                continue
            dg = polyline_dist(gt_px_meta, og)
            d_g = np.minimum(d_g, dg)
            gt_pair_min_dist = min(gt_pair_min_dist, float(dg.min()))
        approach = d_g <= GT_APPROACH_PX
        gt_approach_len = int(approach.sum())
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
                zc = gt_px_meta[s:e + 1].mean(axis=0)
                gt_app_zones.append([float(zc[0]), float(zc[1])])
            az = gt_px_meta[approach]
            if len(az):
                for i in range(0, n, 512):
                    q = pred_px[i:i + 512]
                    dd = np.sqrt(((q[:, None, :] - az[None, :, :]) ** 2).sum(axis=2))
                    in_gt_app[i:i + 512] = dd.min(axis=1) <= JUNC_R
    frac_err_in_gt_app = float((err & in_gt_app).sum() / n_err) if n_err else 0.0
    gt_app_sse_ratio = float((d[in_gt_app] ** 2).sum() / sse) if sse > 0 else 0.0

    # ---- (D) follows another GT curve ----
    on_other_gt = np.zeros(n, bool)
    if len(other_gt_polys):
        d_og = np.full(n, np.inf)
        for og in other_gt_polys:
            if len(og) < 2:
                continue
            d_og = np.minimum(d_og, polyline_dist(pred_px, og))
        on_other_gt = (d > ERR_PX) & (d_og <= 4.0)
    frac_err_on_other_gt = float(on_other_gt.sum() / n_err) if n_err else 0.0
    err_mean = float(d[err].mean()) if n_err else 0.0

    # ---- unified zone ----
    in_zone = in_junc | near_other | in_gt_app
    frac_err_in_zone = float((err & in_zone).sum() / n_err) if n_err else 0.0
    zone_sse_ratio = float((d[in_zone] ** 2).sum() / sse) if sse > 0 else 0.0

    # ---- error runs ----
    runs = []
    if n_err:
        idx = np.where(err)[0]
        start = prev = idx[0]
        run_info = []
        for k in idx[1:]:
            if k - prev > 1:
                run_info.append((start, prev))
                start = k
            prev = k
        run_info.append((start, prev))
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

    # ---- offset profile (uniform offset vs slope/scale) ----
    offset_prof = None
    if len(pred_px) >= 4 and len(gt_px) >= 2:
        gx = gt_px[:, 0]
        go = np.argsort(gx)
        gx, gy = gx[go], gt_px[go, 1]
        x = pred_px[:, 0]
        o = np.argsort(x)
        x, y = x[o], pred_px[o, 1]
        xlo, xhi = max(float(gx[0]), float(x[0])), min(float(gx[-1]), float(x[-1]))
        inside = (x >= xlo) & (x <= xhi)
        if inside.sum() >= 4:
            gyi = np.interp(x[inside], gx, gy)
            dy = y[inside] - gyi
            corr = float(np.corrcoef(x[inside], dy)[0, 1]) if np.std(dy) > 1e-9 else 0.0
            offset_prof = {
                "n": int(inside.sum()),
                "median_abs_dy": round(float(np.median(np.abs(dy))), 3),
                "mean_dy": round(float(dy.mean()), 3),
                "std_dy": round(float(dy.std()), 3),
                "corr_dy_x": round(corr, 3),
            }

    # ---- verdict ----
    def _pre_correct(r) -> bool:
        s = r["start"]
        if s >= 15:
            return float(d[s - 15:s].mean()) <= ERR_PX
        if s > 0:
            return float(d[:s].mean()) <= ERR_PX
        return False

    jump_runs = [r for r in runs if r["follows_other_gt"] and r["len"] >= 5]
    if jump_runs:
        r0 = jump_runs[0]
        if _pre_correct(r0):
            verdict = "crossing_jump"
        else:
            verdict = "ghost" if frac_err_on_other_gt >= 0.6 else "systematic"
    elif (frac_err_in_zone >= 0.5 or zone_sse_ratio >= 0.5) \
            and frac_err_on_other_gt < 0.4 and err_mean <= 25.0:
        verdict = "crossing_merge"
    elif frac_err >= 0.5:
        verdict = "ghost" if frac_err_on_other_gt >= 0.5 else "systematic"
    elif frac_err >= 0.25:
        verdict = "spike"
    elif n_err and n_err <= max(4, int(0.15 * n)):
        verdict = "spike"
    else:
        verdict = "other"

    axis_error = bool(axis_dev > AXIS_DEV_PX and (float(d.mean()) - float(d_meta.mean())) > 1.5
                      and float(d_meta.mean()) < 2.5)
    return {
        "n": n, "n_err": n_err, "frac_err": round(frac_err, 4),
        "frac_err_meta": round(frac_err_meta, 4),
        "mean_px_err": round(float(d.mean()), 4),
        "rmse_px": round(float(np.sqrt(np.mean(d ** 2))), 4),
        "mean_px_err_meta": round(float(d_meta.mean()), 4),
        "axis_dev_px": round(axis_dev, 3),
        "axis_error": axis_error,
        "n_junctions": len(junctions),
        "frac_err_in_junc": round(frac_err_in_junc, 4),
        "junc_sse_ratio": round(junc_sse_ratio, 4),
        "frac_err_near_other": round(frac_err_near_other, 4),
        "other_sse_ratio": round(other_sse_ratio, 4),
        "gt_pair_min_dist_px": round(gt_pair_min_dist, 3) if np.isfinite(gt_pair_min_dist) else None,
        "gt_approach_len_px": gt_approach_len,
        "n_gt_approach_zones": len(gt_app_zones),
        "gt_approach_zones": gt_app_zones,
        "frac_err_in_gt_app": round(frac_err_in_gt_app, 4),
        "gt_app_sse_ratio": round(gt_app_sse_ratio, 4),
        "frac_err_on_other_gt": round(frac_err_on_other_gt, 4),
        "frac_err_in_zone": round(frac_err_in_zone, 4),
        "zone_sse_ratio": round(zone_sse_ratio, 4),
        "n_err_runs": len(runs),
        "n_runs_follow_other": len(jump_runs),
        "err_runs": runs,
        "offset_profile": offset_prof,
        "verdict": verdict,
    }


def curve_rmse_value(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt[np.argsort(gt[:, 0])]
    p = pred[np.argsort(pred[:, 0])]
    x_lo = max(float(g[0, 0]), float(p[0, 0]))
    x_hi = min(float(g[-1, 0]), float(p[-1, 0]))
    inside = (p[:, 0] >= x_lo) & (p[:, 0] <= x_hi)
    if inside.sum() < 2:
        return float("inf")
    gy = np.interp(p[inside, 0], g[:, 0], g[:, 1])
    return float(np.sqrt(np.mean((p[inside, 1] - gy) ** 2)))


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
    for jp in junctions:
        circ = plt.Circle(jp, JUNC_R, color="red", fill=False, ls="--", lw=1.2)
        ax.add_patch(circ)
        ax.plot(jp[0], jp[1], "ro", ms=7, mec="white", mew=0.5)
    for zp in gt_app_zones:
        circ = plt.Circle(zp, JUNC_R, color="magenta", fill=False, ls=":", lw=1.4)
        ax.add_patch(circ)
        ax.plot(zp[0], zp[1], "m^", ms=7, mec="white", mew=0.5)
    ax.plot(gt_px[:, 0], gt_px[:, 1], "-", color="lime", lw=2.2, alpha=0.9, label="GT(pipeline)")
    ax.plot(pred_px[:, 0], pred_px[:, 1], "-", color="blue", lw=1.8, alpha=0.9, label="pred")
    err = d > ERR_PX
    if err.any():
        ax.scatter(pred_px[err, 0], pred_px[err, 1], s=8, c="yellow", marker="o",
                   edgecolors="black", linewidths=0.3, label=f"err>{ERR_PX}px")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, image_rgb.shape[1])
    ax.set_ylim(image_rgb.shape[0], 0)

    zoom_center = None
    if junctions:
        zoom_center = junctions[0]
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

    failures = []
    for r in diag["rows"]:
        for g in r.get("per_gt", []):
            rel = g["rel"]
            if rel > 0.01:
                failures.append({
                    "image": r["image"], "template": r.get("template", "?"),
                    "shape": r.get("shape", "?"), "deg": r.get("deg", ""),
                    "label": g["label"], "pred_chan": g["pred_chan"],
                    "rel": rel, "y_span": g.get("y_span", None),
                    "bucket": "e5p" if (rel > 0.05 or rel == float("inf")) else "e2_5",
                })
    print(f"[chain-px] failures (rel>0.01) from diag.json: {len(failures)} "
          f"({sum(1 for f in failures if f['bucket'] == 'e5p')} e5p, "
          f"{sum(1 for f in failures if f['bucket'] == 'e2_5')} e2-5) "
          f"across {len({f['image'] for f in failures})} images")

    img_path_of = {}
    for f in failures:
        name = f["image"]
        if name in img_path_of:
            continue
        for dd in IMG_DIRS:
            p = os.path.join(ROOT, "data", dd, name)
            if os.path.exists(p):
                img_path_of[name] = p
                break

    os.makedirs(VIS_DIR, exist_ok=True)
    cfg = load_config()
    print(f"[chain-px] checkpoint: {CKPT}")
    segmenter = MultiUNetSegmenter(CKPT, size=int(cfg.get("unet_size", 512)))
    cfg["multi_unet_checkpoint"] = CKPT

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
            meta_px = load_meta_curves_px(stem)

            surv = [c for c, m in enumerate(masks) if m is not None]
            if len(curves) != len(surv):
                raise RuntimeError(f"curve/channel count mismatch: {len(curves)} vs {len(surv)}")

            x0, y0, x1, y1 = structure.plot_bbox
            junctions_of = {}
            for c in surv:
                jl = cluster_junctions(find_branch_points(masks[c]))
                junctions_of[c] = [(float(a + x0), float(b + y0)) for a, b in jl]

            gt_by_label = {lab: pts for lab, pts in gts}

            for f in flist:
                sub = dict(f)
                try:
                    j = f["pred_chan"]
                    if j is None or j >= len(curves):
                        sub.update({"verdict": "missing",
                                    "error": f"pred_chan {j} out of range / no pred matched"})
                        rec["curve_failures"].append(sub)
                        continue
                    ch = surv[j]
                    gt_pts = gt_by_label.get(f["label"])
                    if gt_pts is None or len(gt_pts) < 2:
                        sub.update({"verdict": "missing", "error": "GT not found"})
                        rec["curve_failures"].append(sub)
                        continue
                    gt_px = np.column_stack([
                        x_axis.value_to_pixel(gt_pts[:, 0]),
                        y_axis.value_to_pixel(gt_pts[:, 1]),
                    ]).astype(np.float64)
                    # meta GT pixel chain for the same GT index (order = gts order)
                    gi = next(i for i, (lab, _) in enumerate(gts) if lab == f["label"])
                    gt_px_meta = meta_px[gi] if gi < len(meta_px) else gt_px
                    pred_px = np.asarray(curves[j]["pixel_points"], dtype=np.float64)
                    if len(pred_px) < 8:
                        sub.update({"verdict": "missing", "error": "pred chain too short"})
                        rec["curve_failures"].append(sub)
                        continue
                    junctions = junctions_of[ch]
                    other_masks = [masks[c] for c in surv if c != ch]
                    other_gt_polys_meta = [
                        meta_px[i] for i in range(len(gts)) if i != gi and len(meta_px[i]) >= 2
                    ]
                    ana = analyze_curve(pred_px, gt_px, gt_px_meta, masks[ch], junctions,
                                        other_masks, other_gt_polys_meta, x0, y0)
                    d = polyline_dist(pred_px, gt_px)

                    pred_data = np.asarray(curves[j]["points"], dtype=np.float64)
                    best_gt = {"label": None, "rel": None}
                    for lab2, pts2 in gts:
                        r = curve_rmse_value(pts2, pred_data)
                        if r == float("inf"):
                            continue
                        ysp = float(pts2[:, 1].max() - pts2[:, 1].min())
                        rel2 = r / ysp if ysp > 0 else float("inf")
                        if best_gt["rel"] is None or rel2 < best_gt["rel"]:
                            best_gt = {"label": lab2, "rel": round(rel2, 4)}

                    safe = "".join(ch_ for ch_ in f["label"] if ch_.isalnum() or ch_ in "_- ") or "curve"
                    vis_name = f"{os.path.splitext(name)[0]}_{safe.strip().replace(' ', '_')}.png"
                    vis_path = os.path.join(VIS_DIR, vis_name)
                    title = (f"{name} | {f['label']} | ch={j} | rel={f['rel']:.3f} | "
                             f"{ana['verdict']} | junc={len(junctions)} | ax={ana['axis_error']}")
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if f["bucket"] == "e5p":
                        save_vis(image_rgb, structure, masks, surv, junctions,
                                 pred_px, gt_px, d, title, vis_path,
                                 gt_app_zones=ana["gt_approach_zones"])

                    sub.update({
                        "pred_chan": j, "channel": ch, "rel": f["rel"],
                        "verdict": ana["verdict"],
                        "frac_err": ana["frac_err"],
                        "frac_err_meta": ana["frac_err_meta"],
                        "mean_px_err": ana["mean_px_err"],
                        "rmse_px": ana["rmse_px"],
                        "mean_px_err_meta": ana["mean_px_err_meta"],
                        "axis_dev_px": ana["axis_dev_px"],
                        "axis_error": ana["axis_error"],
                        "n_junctions": ana["n_junctions"],
                        "frac_err_in_junc": ana["frac_err_in_junc"],
                        "junc_sse_ratio": ana["junc_sse_ratio"],
                        "frac_err_near_other": ana["frac_err_near_other"],
                        "other_sse_ratio": ana["other_sse_ratio"],
                        "gt_pair_min_dist_px": ana["gt_pair_min_dist_px"],
                        "gt_approach_len_px": ana["gt_approach_len_px"],
                        "n_gt_approach_zones": ana["n_gt_approach_zones"],
                        "gt_approach_zones": ana["gt_approach_zones"],
                        "frac_err_in_gt_app": ana["frac_err_in_gt_app"],
                        "gt_app_sse_ratio": ana["gt_app_sse_ratio"],
                        "frac_err_on_other_gt": ana["frac_err_on_other_gt"],
                        "frac_err_in_zone": ana["frac_err_in_zone"],
                        "zone_sse_ratio": ana["zone_sse_ratio"],
                        "n_err_runs": ana["n_err_runs"],
                        "n_runs_follow_other": ana["n_runs_follow_other"],
                        "err_runs": ana["err_runs"],
                        "offset_profile": ana["offset_profile"],
                        "best_gt": best_gt,
                        "n_pred_img": len(curves),
                        "vis": os.path.relpath(vis_path, ROOT) if f["bucket"] == "e5p" else None,
                        "error": None,
                    })
                    print(f"[OK ] {name} {f['label']}: ch={j} rel={f['rel']:.4f} "
                          f"fracErr={ana['frac_err']:.2f} onOther={ana['frac_err_on_other_gt']:.2f} "
                          f"zone={ana['frac_err_in_zone']:.2f} minGtPair={ana['gt_pair_min_dist_px']}px "
                          f"axisDev={ana['axis_dev_px']:.1f} -> {ana['verdict']}"
                          + (" [AXIS]" if ana["axis_error"] else ""))
                except Exception as e:
                    sub.update({"verdict": "error",
                                "error": f"{type(e).__name__}: {e}"})
                    print(f"[ERR] {name} {f['label']}: {sub['error']}")
                rec["curve_failures"].append(sub)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[IMG-ERR] {name}: {rec['error']}")
        results.append(rec)

    ok_fails = [sf for r in results for sf in r["curve_failures"]
                if sf.get("error") is None and sf.get("verdict") != "error"]
    verdicts = Counter(sf["verdict"] for sf in ok_fails)
    e5p_fails = [sf for sf in ok_fails if sf["bucket"] == "e5p"]
    e2_fails = [sf for sf in ok_fails if sf["bucket"] == "e2_5"]
    e5p_v = Counter(sf["verdict"] for sf in e5p_fails)
    e2_v = Counter(sf["verdict"] for sf in e2_fails)
    crossing = verdicts["crossing_jump"] + verdicts["crossing_merge"]

    print("\n================ chain-px summary ================")
    print(f"failures analysed: {len(ok_fails)} / {len(failures)}")
    print(f"all verdicts: {dict(verdicts)}")
    print(f"e5p verdicts: {dict(e5p_v)}  (crossing-family {e5p_v['crossing_jump'] + e5p_v['crossing_merge']}/{len(e5p_fails)})")
    print(f"e2-5 verdicts: {dict(e2_v)}")
    if e5p_fails:
        z = [sf["frac_err_in_zone"] for sf in e5p_fails]
        o = [sf["frac_err_on_other_gt"] for sf in e5p_fails]
        ga = [sf["frac_err_in_gt_app"] for sf in e5p_fails]
        md = [sf["gt_pair_min_dist_px"] for sf in e5p_fails if sf["gt_pair_min_dist_px"] is not None]
        ax = [sf["axis_error"] for sf in e5p_fails]
        print(f"e5p zone_frac mean={np.mean(z):.3f} on_other mean={np.mean(o):.3f} "
              f"gt_app mean={np.mean(ga):.3f} minGtPair median={np.median(md):.1f}px "
              f"axis_error={sum(ax)}/{len(ax)}")
    tpl = defaultdict(Counter)
    for sf in ok_fails:
        tpl[sf["verdict"]][sf["template"]] += 1
    for v in sorted(verdicts):
        print(f"  {v}: {dict(tpl[v])}")

    out = {
        "meta": {
            "script": "scripts/diag_chain_px.py",
            "diag_source": DIAG,
            "checkpoint": CKPT,
            "config": {k: cfg.get(k) for k in
                       ("multi_mask_thr", "multi_independent_mask", "multi_min_area",
                        "multi_embed_merge", "multi_truncate_jumps", "multi_refine_radius",
                        "multi_refine_bias_y")},
            "err_px_threshold": ERR_PX,
            "gt_approach_px": GT_APPROACH_PX,
            "axis_dev_px": AXIS_DEV_PX,
        },
        "summary": {
            "n_failures": len(failures),
            "n_images": len(by_img),
            "n_analysed": len(ok_fails),
            "verdicts_all": dict(verdicts),
            "verdicts_e5p": dict(e5p_v),
            "verdicts_e2_5": dict(e2_v),
            "crossing_family_e5p": e5p_v["crossing_jump"] + e5p_v["crossing_merge"],
            "verdict_by_template": {v: dict(c) for v, c in tpl.items()},
        },
        "images": results,
    }
    out_path = os.path.join(OUT_DIR, "diag_chain_px.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n[chain-px] wrote {out_path}")
    print(f"[chain-px] wrote {sum(1 for sf in e5p_fails)} e5p visualizations to {VIS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
