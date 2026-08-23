"""S1: sub-pixel decode comparison on the single-curve synthetic set.

Compares three decode families on top of the SAME skeleton chain and the
SAME U-Net probability map (one forward pass per image):

  * centroid   : existing 6 px window probability-weighted centroid
                 (exactly mci.pipeline.curve_extractor._refine_chain)  [baseline]
  * parabolic  : probability profile sampled along the local curve normal
                 (+-2 px, bilinear), 3-point parabola peak interpolation
  * softargmax : window centroid with weights p^(1/T) (T=1 == centroid,
                 T<1 sharpens toward the mode; T=0.5 / T=0.25 tested)

For every image and every variant we report:
  * data-domain rel_rmse vs GT csv (same definition as evaluate.py)
  * pixel-domain y bias vs the exact fractional GT pixels stored in the
    *_mcg.json sidecar (pixel_points), with point-level slope fits of
    (pred_y - GT_y) against GT pixel y and against log10(GT value).

No src/ code is modified; decode functions live in this script and the
pipeline replicates mci.pipeline.curve_extractor.extract_curves (U-Net path).

Usage:
  python scripts/diag_subpixel_s1.py --data-dir data/eval_phased_500_single \
      --out-dir data/experiments_s1s2 --tag s1
  python scripts/diag_subpixel_s1.py --data-dir data/experiments_s1s2/single_x2 \
      --out-dir data/experiments_s1s2 --tag s2_single_x2 --scale 2
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402

from mci.eval.metrics import curve_metrics  # noqa: E402
from mci.pipeline.chart_structure import detect_structure  # noqa: E402
from mci.pipeline.coordinate_mapper import build_axes  # noqa: E402
from mci.pipeline.curve_extractor import (  # noqa: E402
    _column_centroid,
    _filter_mask_fragments,
    _refine_chain,
    _trace_chain,
)
from mci.pipeline.extractor import load_config  # noqa: E402
from mci.pipeline.segmenter import UNetSegmenter  # noqa: E402
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks  # noqa: E402
from mci.utils import downsample_chain, read_image  # noqa: E402


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------
def _bilinear(prob: np.ndarray, x: float, y: float) -> float:
    h, w = prob.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx = x - x0
    fy = y - y0

    def _g(xi: int, yi: int) -> float:
        xi = min(max(xi, 0), w - 1)
        yi = min(max(yi, 0), h - 1)
        return float(prob[yi, xi])

    v00 = _g(x0, y0)
    v10 = _g(x0 + 1, y0)
    v01 = _g(x0, y0 + 1)
    v11 = _g(x0 + 1, y0 + 1)
    return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy + v11 * fx * fy)


def _parabolic_delta(v0: float, v1: float, v2: float) -> float:
    """3-point parabola peak offset from the center sample."""
    denom = v0 - 2.0 * v1 + v2
    if abs(denom) < 1e-12:
        return 0.0
    return float(0.5 * (v0 - v2) / denom)


def _refine_parabolic(prob: np.ndarray, chain, half: int = 2,
                      min_prob: float = 0.3) -> list:
    """Normal-direction probability-profile parabolic sub-pixel refinement.

    For each skeleton point the local tangent is estimated from the chain
    neighbours; the probability profile is sampled along the normal at
    integer offsets in [-half, half] (bilinear); the peak location is
    refined by a 3-point parabola over [argmax-1, argmax, argmax+1].
    """
    pts = np.asarray(chain, dtype=np.float64)
    n = len(pts)
    out = []
    for i in range(n):
        x, y = pts[i]
        if n == 1:
            out.append((float(x), float(y)))
            continue
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[i] - pts[i - 1]
        else:
            t = pts[i + 1] - pts[i - 1]
        tl = float(np.hypot(t[0], t[1]))
        if tl < 1e-9:
            t = np.array([1.0, 0.0])
            tl = 1.0
        t = t / tl
        nx, ny = t[1], -t[0]
        offs = np.arange(-half, half + 1, dtype=np.float64)
        vals = np.array([_bilinear(prob, x + o * nx, y + o * ny) for o in offs])
        if float(vals.max()) < min_prob:
            out.append((float(x), float(y)))
            continue
        k = int(np.argmax(vals))
        if k == 0 or k == len(offs) - 1:
            # peak at window edge: keep the skeleton point (rare; the window
            # is centered on the mask/skeleton which tracks the prob ridge)
            out.append((float(x), float(y)))
            continue
        delta = _parabolic_delta(vals[k - 1], vals[k], vals[k + 1])
        delta = max(-1.0, min(1.0, delta))
        off = offs[k] + delta
        out.append((float(x + off * nx), float(y + off * ny)))
    return out


def _refine_softargmax(prob: np.ndarray, chain, T: float = 1.0,
                       radius: int = 6, min_prob: float = 0.3) -> list:
    """Window centroid with softmax(log p / T) == p^(1/T) weights.

    T=1 reproduces the plain probability-weighted centroid exactly;
    T<1 sharpens the weights so the estimate moves toward the mode of the
    window (the probability ridge).
    """
    h, w = prob.shape
    out = []
    for x, y in chain:
        x0 = max(0, int(x) - radius)
        x1 = min(w, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(h, int(y) + radius + 1)
        patch = prob[y0:y1, x0:x1]
        if float(patch.max()) < min_prob:
            out.append((float(x), float(y)))
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float64),
                             np.arange(y0, y1, dtype=np.float64))
        wts = patch.astype(np.float64) ** (1.0 / T)
        s = float(wts.sum())
        if s < 1e-12:
            out.append((float(x), float(y)))
            continue
        out.append((float((wts * xs).sum() / s),
                    float((wts * ys).sum() / s)))
    return out


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------
def load_gt_csv(stem: str) -> np.ndarray:
    with open(stem + ".csv", "r", encoding="utf-8") as f:
        rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
    return np.array([[float(r[0]), float(r[1])] for r in rows[1:]])


def load_gt_pixels_mcg(stem: str, scale: float) -> np.ndarray:
    """Exact fractional GT curve pixels from the *_mcg.json sidecar
    (full-image coordinates, scaled by `scale` for upsampled inputs)."""
    with open(stem + "_mcg.json", "r", encoding="utf-8") as f:
        mcg = json.load(f)
    pts = np.asarray(mcg["curves"][0]["pixel_points"], dtype=np.float64)
    return pts * scale


def pixel_bias(pred_full: np.ndarray, gt_px: np.ndarray):
    """pred_full: (N,2) full-image fractional pixels; gt_px: (M,2) GT pixels.
    Returns (stats dict, dy array, gt_py array, gt_data_y array)."""
    p = pred_full[np.argsort(pred_full[:, 0])]
    g = gt_px[np.argsort(gt_px[:, 0])]
    stats = {"n": 0, "mean_y": 0.0, "std_y": 0.0, "rmse_y": 0.0, "median_y": 0.0,
             "mad_y": 0.0, "frac_gt1_y": 0.0, "frac_gt2_y": 0.0,
             "mean_x": 0.0, "std_x": 0.0, "rmse_x": 0.0}
    if len(p) < 3 or len(g) < 3:
        return stats, np.array([]), np.array([]), np.array([])
    inside = (p[:, 0] >= g[0, 0]) & (p[:, 0] <= g[-1, 0])
    if inside.sum() < 3:
        return stats, np.array([]), np.array([]), np.array([])
    px = p[inside, 0]
    py = p[inside, 1]
    gy = np.interp(px, g[:, 0], g[:, 1])
    dy = py - gy
    gx = np.interp(py, g[:, 1], g[:, 0])
    dx = px - gx
    stats.update({
        "n": int(len(dy)),
        "mean_y": float(np.mean(dy)),
        "std_y": float(np.std(dy)),
        "rmse_y": float(np.sqrt(np.mean(dy ** 2))),
        "median_y": float(np.median(dy)),
        "mad_y": float(np.median(np.abs(dy - np.median(dy)))),
        "frac_gt1_y": float(np.mean(np.abs(dy) > 1.0)),
        "frac_gt2_y": float(np.mean(np.abs(dy) > 2.0)),
        "mean_x": float(np.mean(dx)),
        "std_x": float(np.std(dx)),
        "rmse_x": float(np.sqrt(np.mean(dx ** 2))),
    })
    return stats, dy, gy, np.array([])


def ols_slope(x: np.ndarray, y: np.ndarray) -> dict:
    """y = intercept + slope * x (least squares)."""
    if len(x) < 4 or float(np.ptp(x)) < 1e-9:
        return {"slope": None, "intercept": None, "r2": None, "n": int(len(x))}
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"slope": float(a), "intercept": float(b), "r2": float(r2),
            "n": int(len(x))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_decoders():
    return {
        "centroid": lambda prob, chain: _refine_chain(prob, chain, radius=6),
        "parabolic": lambda prob, chain: _refine_parabolic(prob, chain),
        "softargmax_T1.0": lambda prob, chain: _refine_softargmax(prob, chain, T=1.0),
        "softargmax_T0.5": lambda prob, chain: _refine_softargmax(prob, chain, T=0.5),
        "softargmax_T0.25": lambda prob, chain: _refine_softargmax(prob, chain, T=0.25),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/eval_phased_500_single")
    ap.add_argument("--out-dir", default="data/experiments_s1s2")
    ap.add_argument("--model", default="models/checkpoints/unet_curve.pt")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="pixel scale factor of this dataset relative to the "
                         "original sidecars (2 for the 2x-upsampled set)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config()
    seg = UNetSegmenter(args.model, size=args.size)
    decoders = build_decoders()
    decoder_names = list(decoders)

    images = sorted(glob.glob(os.path.join(args.data_dir, "*.png")))
    images = [p for p in images if not os.path.basename(p).endswith("_mask.png")]
    if args.limit:
        images = images[: args.limit]
    print(f"[s1] {len(images)} images, decoders={decoder_names}")

    rows = []
    n_fallback = 0
    # pooled point-level accumulators per decoder: dy, gt_py, image tag
    pool = {dn: {"dy": [], "gt_py": [], "y_kind": [], "line_width": [],
                 "template": [], "log10_y": [], "img_id": []} for dn in decoder_names}

    for idx, img_path in enumerate(images):
        stem = os.path.splitext(img_path)[0]
        name = os.path.basename(img_path)
        meta = {}
        mp = stem + "_meta.json"
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
        y_kind = meta.get("y_kind", "?")
        lw = str(meta.get("line_width", "?"))
        tpl = meta.get("template_id", "?")
        row = {
            "image": name,
            "x_kind": meta.get("x_kind", "?"),
            "y_kind": y_kind,
            "line_width": lw,
            "template": tpl,
            "degradations": meta.get("degradations", []),
            "image_size": meta.get("image_size", []),
            "status": "ok",
            "per_decoder": {},
        }
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
            x0, y0, x1, y1 = structure.plot_bbox
            plot_w, plot_h = x1 - x0 + 1, y1 - y0 + 1

            prob = seg.prob_full(img)
            region = prob[y0:y1 + 1, x0:x1 + 1]
            if float(region.max()) < 0.5:
                raise RuntimeError("segmentation found no curve in plot region")
            mask01 = (region > 0.5).astype(np.uint8)
            mask01 = _filter_mask_fragments(mask01, plot_w, plot_h)
            if int(mask01.sum()) < 16:
                raise RuntimeError("segmentation mask too small")
            skel = skeletonize(mask01.astype(bool)).astype(np.uint8)
            chain = _trace_chain(skel)
            if chain is not None:
                xs = [c[0] for c in chain]
                if (max(xs) - min(xs) + 1) < 0.7 * plot_w:
                    chain = None
            fallback = chain is None
            if fallback:
                n_fallback += 1
                cc = _column_centroid(region)
                chains = {dn: list(cc) for dn in decoder_names}
            else:
                chains = {dn: decoders[dn](region, chain) for dn in decoder_names}

            gt = load_gt_csv(stem)
            gt_px = load_gt_pixels_mcg(stem, args.scale)
            for dn in decoder_names:
                ch = downsample_chain(chains[dn], int(cfg.get("max_points", 2000)))
                if len(ch) < 4:
                    row["per_decoder"][dn] = {"rel_rmse": None, "rmse": None,
                                              "pass": False, "x_coverage": 0.0,
                                              "n_pred": len(ch), "error": "chain too short"}
                    continue
                points = [(x_axis.pixel_to_value(x0 + px), y_axis.pixel_to_value(y0 + py))
                          for px, py in ch]
                m = curve_metrics(gt, points)
                pred_full = np.array([(x0 + px, y0 + py) for px, py in ch], dtype=np.float64)
                bias, dy, gy_px, _ = pixel_bias(pred_full, gt_px)
                rel = m["rel_rmse"]
                row["per_decoder"][dn] = {
                    "rel_rmse": float(rel) if np.isfinite(rel) else None,
                    "rmse": float(m["rmse"]) if np.isfinite(m["rmse"]) else None,
                    "pass": bool(np.isfinite(rel) and rel <= 0.01),
                    "x_coverage": m["x_coverage"],
                    "n_pred": m["n_pred"],
                    "max_abs_err": float(m["max_abs_err"]) if np.isfinite(m["max_abs_err"]) else None,
                    "px_bias": bias,
                }
                if len(dy):
                    # data-domain GT value at the pred x (for log-scale slope)
                    pdata = np.asarray(points, dtype=np.float64)
                    gv = np.interp(pdata[:, 0], gt[:, 0], gt[:, 1])
                    gv = gv[:len(dy)]
                    log10v = np.log10(np.maximum(gv, 1e-12))
                    pool[dn]["dy"].append(dy)
                    pool[dn]["gt_py"].append(gy_px)
                    pool[dn]["log10_y"].append(log10v)
                    pool[dn]["y_kind"].extend([y_kind] * len(dy))
                    pool[dn]["line_width"].extend([lw] * len(dy))
                    pool[dn]["template"].extend([tpl] * len(dy))
                    pool[dn]["img_id"].extend([idx] * len(dy))
            row["fallback"] = fallback
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if (idx + 1) % 50 == 0 or idx + 1 == len(images):
            print(f"[s1] {idx + 1}/{len(images)}  {name}  "
                  f"fallback={fallback}  status={row['status']}")

    # ---------------- summaries ----------------
    summary = summarize_single(rows, decoder_names)
    cmp_path = os.path.join(args.out_dir, f"{args.tag}_decode_compare.json")
    with open(cmp_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"data_dir": args.data_dir, "model": args.model,
                            "size": args.size, "scale": args.scale,
                            "n_images": len(images), "decoders": decoder_names},
                   "summary": summary, "images": rows}, f, indent=1, ensure_ascii=False)

    bias_analysis = build_bias_analysis(pool, decoder_names)
    bias_path = os.path.join(args.out_dir, f"{args.tag}_bias_analysis.json")
    with open(bias_path, "w", encoding="utf-8") as f:
        json.dump(bias_analysis, f, indent=1, ensure_ascii=False)

    print("\n================ summary ================")
    for dn in decoder_names:
        s = summary[dn]
        print(f"  [{dn}] pass_rate_1pct={s['pass_rate_1pct']:.4f} "
              f"median_rel={s['median_rel']:.5f} mean_rel={s['mean_rel']:.5f} "
              f"p90_rel={s['p90_rel']:.5f} buckets={s['buckets']} "
              f"bias_y_mean={s['px_bias']['image_mean_mean']:+.3f}px "
              f"bias_y_rmse={s['px_bias']['image_rmse_mean']:.3f}px")
    print(f"fallback (column-centroid) images: {n_fallback}")
    print(f"wrote: {cmp_path}\n       {bias_path}")
    return 0


def summarize_single(rows, decoder_names) -> dict:
    out = {}
    for dn in decoder_names:
        rels = []
        for r in rows:
            if r["status"] != "ok":
                continue
            v = r["per_decoder"].get(dn, {}).get("rel_rmse")
            if v is not None and np.isfinite(v):
                rels.append(v)
        biases = [r["per_decoder"][dn]["px_bias"] for r in rows
                  if r["status"] == "ok"
                  and r["per_decoder"].get(dn, {}).get("px_bias", {}).get("n", 0) > 0]
        img_means = [b["mean_y"] for b in biases]
        img_rmse = [b["rmse_y"] for b in biases]
        out[dn] = {
            "pass_rate_1pct": float(np.mean(np.asarray(rels) <= 0.01)) if rels else float("nan"),
            "median_rel": float(np.median(rels)) if rels else float("nan"),
            "mean_rel": float(np.mean(rels)) if rels else float("nan"),
            "p90_rel": float(np.percentile(rels, 90)) if rels else float("nan"),
            "max_rel": float(np.max(rels)) if rels else float("nan"),
            "n_ok": len(rels),
            "n_images": len(rows),
            "buckets": {
                "ok": int(sum(1 for v in rels if v <= 0.01)),
                "e1_2": int(sum(1 for v in rels if 0.01 < v <= 0.02)),
                "e2_5": int(sum(1 for v in rels if 0.02 < v <= 0.05)),
                "e5p": int(sum(1 for v in rels if v > 0.05)),
            },
            "px_bias": {
                "n_images": len(img_means),
                "image_mean_mean": float(np.mean(img_means)) if img_means else float("nan"),
                "image_mean_std": float(np.std(img_means)) if img_means else float("nan"),
                "image_mean_median": float(np.median(img_means)) if img_means else float("nan"),
                "image_rmse_mean": float(np.mean(img_rmse)) if img_rmse else float("nan"),
                "frac_images_abs_bias_gt1px": float(
                    np.mean(np.abs(np.asarray(img_means)) > 1.0)) if img_means else float("nan"),
            },
        }
    return out


def build_bias_analysis(pool, decoder_names) -> dict:
    """Pooled point-level bias stats and slope fits.

    slope_vs_pixel_y : dy = a + b * GT_pixel_y  (b ~ 0  -> constant offset)
    slope_vs_log10   : dy = a + b * log10(GT value)  (log axes)
    within_image     : dy - mean_i  vs  GT_pixel_y - mean_i (removes the
                       per-image constant offset; exposes position-dependent
                       bias, e.g. window-centroid drift on steep segments)
    """
    out = {"per_decoder": {}}

    def _fit_entry(dy, gt_py, log10v):
        dy = np.asarray(dy, dtype=np.float64)
        gt_py = np.asarray(gt_py, dtype=np.float64)
        log10v = np.asarray(log10v, dtype=np.float64)
        if len(dy) < 8:
            return None
        entry = {
            "n_points": int(len(dy)),
            "point_mean": float(np.mean(dy)),
            "point_std": float(np.std(dy)),
            "point_rmse": float(np.sqrt(np.mean(dy ** 2))),
            "point_median": float(np.median(dy)),
            "frac_abs_gt1px": float(np.mean(np.abs(dy) > 1.0)),
            "frac_abs_gt2px": float(np.mean(np.abs(dy) > 2.0)),
            "slope_vs_pixel_y": ols_slope(gt_py, dy),
            "slope_vs_log10": ols_slope(log10v, dy),
        }
        return entry

    for dn in decoder_names:
        dy_all = np.concatenate(pool[dn]["dy"]) if pool[dn]["dy"] else np.array([])
        gp_all = np.concatenate(pool[dn]["gt_py"]) if pool[dn]["gt_py"] else np.array([])
        lg_all = np.concatenate(pool[dn]["log10_y"]) if pool[dn]["log10_y"] else np.array([])
        im_all = np.asarray(pool[dn]["img_id"]) if pool[dn]["img_id"] else np.array([])
        yk = np.asarray(pool[dn]["y_kind"])
        lw = np.asarray(pool[dn]["line_width"])
        tpl = np.asarray(pool[dn]["template"])
        entry = _fit_entry(dy_all, gp_all, lg_all)
        if entry is not None:
            entry["n_images"] = int(len(np.unique(im_all)))
            # within-image detrended slope: removes the per-image constant
            # offset; a non-zero slope here means the bias varies along the
            # curve (e.g. window-centroid drift on steep segments)
            dy_c = np.empty_like(dy_all)
            gp_c = np.empty_like(gp_all)
            for iid in np.unique(im_all):
                m = im_all == iid
                if m.sum() >= 4:
                    dy_c[m] = dy_all[m] - dy_all[m].mean()
                    gp_c[m] = gp_all[m] - gp_all[m].mean()
            entry["within_image_slope_vs_pixel_y"] = ols_slope(gp_c, dy_c)
        out["per_decoder"][dn] = entry or {"n_points": 0}

        # groups
        groups = {}
        for gname, arr in (("y_kind", yk), ("line_width", lw), ("template", tpl)):
            gd = {}
            for gval in sorted(set(arr.tolist())):
                m = arr == gval
                e = _fit_entry(dy_all[m], gp_all[m], lg_all[m])
                gd[str(gval)] = e or {"n_points": 0}
            groups[gname] = gd
        out["per_decoder"][dn]["groups"] = groups

    # within-image detrended slope: requires image ids; approximate using the
    # concatenation boundaries is not possible here, so compute it separately
    # in the per-image loop output (per-image means) -- see decode_compare.
    return out


if __name__ == "__main__":
    sys.exit(main())
