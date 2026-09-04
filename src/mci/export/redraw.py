"""Redraw extracted curves as a clean matplotlib figure for comparison.

Reads the full ``result.json`` (including axis specs and per-curve data points)
and produces a fresh PNG plot that mirrors the original chart's curve traces
without the image background, so the user can visually compare extracted data
against the original image.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .reconstruct import clean_curve_points


def redraw_from_json(result_path: str | Path, out_png: str | Path,
                     figsize: tuple = (6.4, 4.8), dpi: int = 150,
                     clean: bool = True, clean_kwargs: dict | None = None) -> str:
    """Redraw extracted curves as a clean matplotlib figure.

    Parameters
    ----------
    result_path
        Path to the extraction ``result.json``.
    out_png
        Output PNG path.
    figsize, dpi
        Figure dimensions (passed to ``plt.subplots``).
    clean
        Apply the robust reconstruction (Hampel spike rejection + light
        LOWESS) to each curve's points before drawing, so tracing/fallback
        artifacts ("错点") do not distort the redrawn shape.  This only affects
        the rendering — it never alters the extracted ``points`` in the result.
    clean_kwargs
        Optional overrides passed to ``clean_curve_points`` (e.g.
        ``{"monotone_y": True}`` only when monotonicity is physically certain,
        ``{"lo_frac": 0.2}`` for stronger smoothing).

    Returns
    -------
    str
        The ``out_png`` path on success.

    Raises
    ------
    FileNotFoundError
        If ``result_path`` does not exist.
    KeyError
        If the JSON lacks required axis/curve keys.
    """
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)

    x_axis = data["x_axis"]
    y_axis = data["y_axis"]
    curves = data["curves"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Determine axis type from the export's "kind" field
    x_kind = x_axis.get("kind", "linear")
    y_kind = y_axis.get("kind", "linear")
    x_is_log = x_kind == "log"
    y_is_log = y_kind == "log"

    # Collect all data points to compute axis limits
    all_x = []
    all_y = []

    for curve in curves:
        pts = np.array(curve["points"], dtype=float)  # (N, 2)
        if pts.ndim == 2 and pts.shape[1] == 2:
            x, y = pts[:, 0], pts[:, 1]
            # Filter out NaN / Inf
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 2:
                continue
            if clean:
                kwargs = dict(clean_kwargs or {})
                kwargs.setdefault("x_is_log", x_is_log)
                kwargs.setdefault("y_is_log", y_is_log)
                cleaned = clean_curve_points(list(zip(x.tolist(), y.tolist())),
                                             **kwargs)
                x = np.asarray([p[0] for p in cleaned], dtype=float)
                y = np.asarray([p[1] for p in cleaned], dtype=float)
                if len(x) < 2:
                    continue
            all_x.append(x)
            all_y.append(y)
            color = curve.get("color")
            if color and len(color) == 3:
                c = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            else:
                c = None
            label = curve.get("legend_label") or curve.get("name", "")
            ax.plot(x, y, color=c, lw=1.5, label=label or None)

    if not all_x:
        ax.text(0.5, 0.5, "No valid curve data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        fig.savefig(str(out_png), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return str(out_png)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    # Axis scale
    if x_kind == "log":
        ax.set_xscale("log")
    if y_kind == "log":
        ax.set_yscale("log")

    # Axis limits (with padding)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    if x_kind == "log":
        x_lo = 10.0 ** np.floor(np.log10(x_min))
        x_hi = 10.0 ** (np.ceil(np.log10(x_max)) + 0.1)
        ax.set_xlim(x_lo, x_hi)
    else:
        x_pad = max((x_max - x_min) * 0.05, 1e-6)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if y_kind == "log":
        y_lo = 10.0 ** np.floor(np.log10(y_min))
        y_hi = 10.0 ** (np.ceil(np.log10(y_max)) + 0.1)
        ax.set_ylim(y_lo, y_hi)
    else:
        y_pad = max((y_max - y_min) * 0.06, 1e-6)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Labels (from the export if available)
    ax.set_xlabel(x_axis.get("role", "x").upper())
    ax.set_ylabel(y_axis.get("role", "y").upper())

    # Grid
    ax.grid(True, which="major", alpha=0.35, ls="--", lw=0.7, color="0.3")

    # Legend (if multiple curves or labels present)
    n_plotted = len(all_x)  # placeholder; we track actual plotted curves
    handles = [h for h in ax.get_legend_handles_labels()[0] if h]
    if len(handles) > 1:
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return str(out_png)