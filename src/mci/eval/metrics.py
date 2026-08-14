"""Quantitative evaluation metrics (baseline set).

Primary acceptance metric (see Prompt.md):
  curve data-point RMSE <= 1% of the axis full scale.

``curve_metrics`` compares a predicted point set against the ground-truth
point set: GT is linearly interpolated at the predicted x positions (within
the overlapping x-range), and the y error is summarized.  The x-range
coverage term additionally penalizes predictions that only cover part of the
curve.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def curve_metrics(
    gt: Sequence[Tuple[float, float]],
    pred: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """Return {n_pred, x_coverage, rmse, rel_rmse, max_abs_err}."""
    g = np.asarray(gt, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if len(g) < 2 or len(p) < 2:
        return {"n_pred": len(p), "x_coverage": 0.0, "rmse": float("nan"),
                "rel_rmse": float("nan"), "max_abs_err": float("nan")}

    # sort by x, drop duplicate x (keep first)
    g = g[np.argsort(g[:, 0])]
    p = p[np.argsort(p[:, 0])]
    mask = np.ones(len(p), dtype=bool)
    mask[1:] = np.diff(p[:, 0]) > 1e-12
    p = p[mask]

    gx_span = g[-1, 0] - g[0, 0]
    if gx_span <= 0:
        return {"n_pred": len(p), "x_coverage": 0.0, "rmse": float("nan"),
                "rel_rmse": float("nan"), "max_abs_err": float("nan")}

    x_lo = max(g[0, 0], p[0, 0])
    x_hi = min(g[-1, 0], p[-1, 0])
    coverage = max(0.0, (x_hi - x_lo) / gx_span)

    inside = (p[:, 0] >= x_lo) & (p[:, 0] <= x_hi)
    if inside.sum() == 0:
        return {"n_pred": len(p), "x_coverage": coverage, "rmse": float("nan"),
                "rel_rmse": float("nan"), "max_abs_err": float("nan")}

    gt_y = np.interp(p[inside, 0], g[:, 0], g[:, 1])
    err = p[inside, 1] - gt_y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    y_span = float(g[:, 1].max() - g[:, 1].min())
    rel = rmse / y_span if y_span > 0 else float("nan")
    return {
        "n_pred": int(len(p)),
        "x_coverage": float(coverage),
        "rmse": rmse,
        "rel_rmse": rel,
        "max_abs_err": float(np.max(np.abs(err))),
    }


def summarize(rows: Sequence[Dict]) -> Dict:
    """Aggregate per-image metrics into a summary dict."""
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("rel_rmse", np.nan))]
    rel = np.array([r["rel_rmse"] for r in ok])
    return {
        "n_images": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "rel_rmse_mean": float(rel.mean()) if len(rel) else float("nan"),
        "rel_rmse_median": float(np.median(rel)) if len(rel) else float("nan"),
        "rel_rmse_p90": float(np.percentile(rel, 90)) if len(rel) else float("nan"),
        "rel_rmse_max": float(rel.max()) if len(rel) else float("nan"),
        "pass_rate_1pct": float(np.mean(rel <= 0.01)) if len(rel) else float("nan"),
    }
