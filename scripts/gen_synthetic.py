"""Synthetic single-curve chart generator with exact ground truth.

Self-contained data factory for the baseline (the official
materials-curve-dataset-platform stays the production data source; see
README "Data").  Generates single-plot, single-curve charts covering:

  * creep-like / power / sigmoid / relaxation / log curves
  * linear & log axes (all 4 combinations)
  * paper-like and experiment-like styles (serif fonts, grids, spines on/off)
  * optional single-entry legend (inside or outside the plot)
  * degradations: JPEG, downscale round-trip, blur, brightness/contrast, noise

Ground truth is exact: the figure is rendered at fig.dpi == save dpi, so
matplotlib display coordinates map 1:1 onto pixels (tight bbox accounted
for via fig.bbox_inches), and a self-check verifies that GT curve points
really map to dark pixels in the saved image.

Outputs per image ``<stem>``:
  <stem>.png             final (possibly degraded) image
  <stem>.csv             GT curve data coordinates (x,y)
  <stem>_meta.json       axis kinds/ranges, tick values, style, degradations
  <stem>_labels.json     GT text boxes (PaddleOCR quad format) for the stub
                         OCR backend (deterministic tests / OCR ablation)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402

# ---------------------------------------------------------------------------
# Curve families (all strictly positive on t in [t0, T])
# ---------------------------------------------------------------------------
def _creep(t, p):
    y = p["c0"] + p["a"] * (1 - np.exp(-t / p["tau"])) + p["b"] * t + p.get("k", 0.0) * t ** 3
    return y


def _power(t, p):
    return p["a"] * t ** p["b"]


def _sigmoid(t, p):
    return p["d"] + p["a"] / (1 + np.exp(-p["k"] * (t - p["c"])))


def _relax(t, p):
    return p["a"] * np.exp(-t / p["tau"]) + p["d"]


def _logcurve(t, p):
    return p["d"] + p["a"] * np.log(1 + p["b"] * t)


FUNCS = {"creep": _creep, "power": _power, "sigmoid": _sigmoid,
         "relax": _relax, "logcurve": _logcurve}

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#000000", "#3b3b3b", "#006400", "#8b0000",
]

X_LABELS = ["Time (s)", "Time (h)", "Time (min)", "t / s", "Time (log)", "Time / s"]
Y_LABELS = ["Strain (%)", "Creep strain (%)", "Stress (MPa)", "Displacement (mm)",
            "Strain / %", "Creep strain / %"]
TITLES = ["Creep curve of sample", "Stress relaxation test", "Creep test @ 650 C",
          "Strain evolution", "Creep behaviour"]
LEGEND_TEXTS = ["Creep curve", "Sample A", "Experiment", "Data", "Stress relaxation"]


def sample_config(rng) -> dict:
    T = float(rng.choice([100.0, 200.0, 500.0, 1000.0, 2000.0]))
    x_kind = rng.choice(["linear"] * 3 + ["log"] * 2)
    y_kind = rng.choice(["linear"] * 3 + ["log"] * 2)
    if x_kind == "log":
        t0 = float(rng.uniform(0.01, 1.0))
        T = t0 * 10 ** float(rng.uniform(1.2, 3.2))
    else:
        t0 = 0.0

    fn = rng.choice(list(FUNCS))
    if fn == "creep":
        params = dict(c0=rng.uniform(0, 3), a=rng.uniform(1, 10),
                      tau=T * rng.uniform(0.05, 0.4), b=rng.uniform(0.5, 5) / T)
        if rng.random() < 0.25:
            params["k"] = rng.uniform(0.0, 0.6) / T ** 3
    elif fn == "power":
        params = dict(a=rng.uniform(0.05, 2.0), b=rng.uniform(0.3, 2.2))
    elif fn == "sigmoid":
        params = dict(d=rng.uniform(0.5, 2), a=rng.uniform(1, 8),
                      c=T * rng.uniform(0.2, 0.8), k=rng.uniform(2, 10) / T)
    elif fn == "relax":
        params = dict(a=rng.uniform(2, 8), d=rng.uniform(0.5, 2),
                      tau=T * rng.uniform(0.05, 0.35))
    else:
        params = dict(d=rng.uniform(0, 2), a=rng.uniform(0.5, 3),
                      b=rng.uniform(2, 50) / T)

    return dict(
        T=T, t0=t0, x_kind=x_kind, y_kind=y_kind, fn=fn, params=params,
        dpi=int(rng.choice([150, 200, 300])),
        figsize=(round(rng.uniform(4.2, 6.6), 1), round(rng.uniform(3.0, 4.6), 1)),
        color=rng.choice(PALETTE),
        lw=round(rng.uniform(1.0, 2.6), 2),
        ls=rng.choice(["solid"] * 6 + ["dashed"]),
        grid=rng.random() < 0.55,
        grid_alpha=round(rng.uniform(0.25, 0.65), 2),
        spines=rng.choice(["box"] * 6 + ["lshape"] * 4),
        serif=rng.random() < 0.35,
        legend=rng.random() < 0.45,
        title=rng.random() < 0.3,
        x_label=rng.choice(X_LABELS),
        y_label=rng.choice(Y_LABELS),
        seed=0,
    )


def render_chart(cfg: dict, rng) -> tuple:
    """Render the chart; returns (image BGR, gt dict with exact pixel mapping)."""
    with plt.rc_context({
        "font.family": "serif" if cfg["serif"] else "sans-serif",
        "axes.formatter.use_mathtext": False,
    }):
        fig, ax = plt.subplots(figsize=cfg["figsize"], dpi=cfg["dpi"])
        T, t0 = cfg["T"], cfg["t0"]
        t = np.linspace(t0 if t0 > 0 else 1e-6, T, 400)
        y = FUNCS[cfg["fn"]](t, cfg["params"])

        # make log-y ranges span at least ~1.2 decades
        if cfg["y_kind"] == "log":
            span = np.log10(y.max() / y.min())
            if span < 1.2:
                y = y * 10 ** (1.2 - span)

        ax.plot(t, y, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"], label="__curve__")

        if cfg["x_kind"] == "log":
            ax.set_xscale("log")
        if cfg["y_kind"] == "log":
            ax.set_yscale("log")

        # axis limits: linear -> tight-ish; log -> window of >= 2 decades so
        # that at least 3 decade tick labels are visible (the automatic
        # linear/log judgement needs >= 3 valued ticks to discriminate)
        if cfg["x_kind"] == "log":
            x_lo = 10.0 ** np.floor(np.log10(t0))
            x_hi = max(10.0 ** (np.floor(np.log10(t0)) + 3.0), T * 1.05)
            ax.set_xlim(x_lo, x_hi)
        else:
            ax.set_xlim(t0, T)
        ymin, ymax = float(y.min()), float(y.max())
        if cfg["y_kind"] == "log":
            y_lo = 10.0 ** np.floor(np.log10(ymin))
            y_hi = max(10.0 ** (np.floor(np.log10(ymin)) + 3.0), ymax * 1.1)
            ax.set_ylim(y_lo, y_hi)
        else:
            pad = (ymax - ymin) * 0.06
            ax.set_ylim(ymin - pad, ymax + pad)

        for axis_name in ("x", "y"):
            if cfg[f"{axis_name}_kind"] == "linear":
                getattr(ax, f"ticklabel_format")(axis=axis_name, useOffset=False, style="plain")
            # force >= 3 labeled ticks
            loc = ax.xaxis if axis_name == "x" else ax.yaxis
            labels = [lb.get_text() for lb in loc.get_ticklabels() if lb.get_text()]
            if len(labels) < 3 and cfg[f"{axis_name}_kind"] == "linear":
                lim = getattr(ax, f"get_{axis_name}lim")()
                loc.set_major_locator(ticker.MultipleLocator((lim[1] - lim[0]) / 5))

        if cfg["spines"] == "lshape":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        if cfg["grid"]:
            ax.grid(True, which="major", alpha=cfg["grid_alpha"], ls="--", lw=0.7, color="0.3")

        legend = None
        if cfg["legend"]:
            if rng.random() < 0.5:
                ax.legend([rng.choice(LEGEND_TEXTS)],
                          loc=rng.choice(["upper left", "upper right", "lower right", "lower left"]),
                          framealpha=0.9)
            else:
                ax.legend([rng.choice(LEGEND_TEXTS)],
                          loc="upper left", bbox_to_anchor=(1.02, 0.98), framealpha=0.9)
        if cfg["title"]:
            ax.set_title(rng.choice(TITLES))
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel(cfg["y_label"])

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Pixel space: use the canvas buffer directly — by construction its
        # pixels ARE the display coordinate space of transData (unlike
        # savefig, which re-renders and can shift by sub-pixel/rounding
        # amounts).  Crop to the ink bbox (+pad) for a paper-like tight
        # figure; the crop offset keeps all GT pixel positions exact.
        rgba = np.asarray(fig.canvas.buffer_rgba())
        gray_full = rgba[:, :, :3].mean(axis=2)
        ink_full = gray_full < 235
        ys, xs = np.nonzero(ink_full)
        pad = 8
        x0c = max(0, int(xs.min()) - pad)
        y0c = max(0, int(ys.min()) - pad)
        x1c = min(rgba.shape[1], int(xs.max()) + pad)
        y1c = min(rgba.shape[0], int(ys.max()) + pad)
        img_bgr = cv2.cvtColor(rgba[y0c:y1c, x0c:x1c, :3], cv2.COLOR_RGB2BGR)

        def to_px(dx, dy):
            """display (y up) -> cropped image pixel (y down)."""
            return (round(dx - x0c), round(rgba.shape[0] - dy - y0c))

        # ---- curve mask ground truth ----
        # Drawn SOLID at the rendered line width, even for dashed curves:
        # the segmentation model learns to complete dash gaps.
        lw_px = max(2, int(round(cfg["lw"] * cfg["dpi"] / 72.0)))
        mask_full = np.zeros((rgba.shape[0], rgba.shape[1]), np.uint8)
        pts_full = [(round(dx), round(rgba.shape[0] - dy))
                    for dx, dy in (ax.transData.transform((tx, ty))
                                   for tx, ty in zip(t[::2], y[::2]))]
        for a, b in zip(pts_full[:-1], pts_full[1:]):
            cv2.line(mask_full, a, b, 255, lw_px)
        mask_img = mask_full[y0c:y1c, x0c:x1c]

        # ---- ground truth ----
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # LogLocator generates ticks for a padded decade range; keep only
        # ticks inside the actual view interval.
        def _in_view(v, lo, hi):
            return v >= lo * (1 - 1e-9) and v <= hi * (1 + 1e-9)

        x_tick_vals = [float(v) for v in ax.get_xticks() if _in_view(v, *xlim)]
        y_tick_vals = [float(v) for v in ax.get_yticks() if _in_view(v, *ylim)]

        gt = {
            "x_kind": cfg["x_kind"], "y_kind": cfg["y_kind"],
            "x_range": [float(xlim[0]), float(xlim[1])],
            "y_range": [float(ylim[0]), float(ylim[1])],
            "x_tick_values": [float(v) for v in x_tick_vals],
            "y_tick_values": [float(v) for v in y_tick_vals],
            "curve_px": [to_px(*ax.transData.transform((tx, ty))) for tx, ty in
                         zip(t[::20], y[::20])],
            "legend": cfg["legend"],
            "function": cfg["fn"],
            "params": cfg["params"],
            "color": cfg["color"],
            "dpi": cfg["dpi"],
            "figsize": cfg["figsize"],
            "degradations": [],
        }

        # stub-OCR sidecar: tick label text boxes, anchored at the TICK MARK
        # positions (transData) rather than at window extents, so the stub is
        # consistent across matplotlib versions; normalized text.
        W, H = img_bgr.shape[1], img_bgr.shape[0]

        def _norm_text(s: str) -> str:
            return (s.replace("$", "").replace("{", "").replace("}", "")
                     .replace("\\mathdefault", "").replace("\\times", "x"))

        labels = []
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        label_h = max(14, int(cfg["dpi"] * 0.16))  # approx label height in px

        def _box_from_center(cx, cy, w_px):
            hw = w_px // 2
            hh = label_h // 2
            return [[cx - hw, cy - hh], [cx + hw, cy - hh],
                    [cx + hw, cy + hh], [cx - hw, cy + hh]]

        x_tick_vals = [float(v) for v in ax.get_xticks() if _in_view(v, *xlim)]
        for v in x_tick_vals:
            mx, my = ax.transData.transform((v, ylim[0]))
            px, py = to_px(mx, my)
            if 0 <= px < W:
                labels.append({"box": _box_from_center(px, py + label_h // 2 + 6, 48),
                               "text": f"{v:g}", "score": 1.0})
        y_tick_vals = [float(v) for v in ax.get_yticks() if _in_view(v, *ylim)]
        for v in y_tick_vals:
            mx, my = ax.transData.transform((xlim[0], v))
            px, py = to_px(mx, my)
            if 0 <= py < H:
                labels.append({"box": _box_from_center(px - label_h // 2 - 6, py, 48),
                               "text": f"{v:g}", "score": 1.0})
        plt.close(fig)
        return img_bgr, mask_img, gt, labels, t, y


def degrade(img_bgr: np.ndarray, rng) -> tuple:
    """Apply a random subset of degradations; returns (image, applied list)."""
    applied = []
    img = img_bgr
    if rng.random() < 0.5:
        q = int(rng.uniform(55, 92))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        applied.append(f"jpeg_q{q}")
    if rng.random() < 0.4:
        s = rng.uniform(0.72, 0.92)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        applied.append(f"resize_{s:.2f}")
    if rng.random() < 0.3:
        sigma = rng.uniform(0.3, 1.0)
        img = cv2.GaussianBlur(img, (0, 0), sigma)
        applied.append(f"blur_{sigma:.2f}")
    if rng.random() < 0.4:
        g = rng.uniform(0.85, 1.15)
        b = rng.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
        applied.append(f"bc_{g:.2f}_{b:.0f}")
    if rng.random() < 0.2:
        mask = rng.random(img.shape[:2])
        img = img.copy()
        img[mask < 0.0004] = 0
        img[mask > 1 - 0.0004] = 255
        applied.append("snp")
    return img, applied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/synthetic", help="output directory")
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-idx", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)

    n_fail = 0
    for i in range(args.count):
        idx = args.start_idx + i
        stem = os.path.join(args.out_dir, f"img_{idx:04d}")
        cfg = sample_config(rng)
        try:
            img, mask_img, gt, labels, t, y = render_chart(cfg, rng)
        except Exception:
            print(f"[{idx}] RENDER FAILED:\n{traceback.format_exc()}")
            n_fail += 1
            continue

        # self-check: GT curve points must be near ink in the saved image
        # (dilated mask tolerates dash gaps and anti-aliasing)
        gray_ck = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink_ck = (gray_ck < 235).astype(np.uint8)
        ink_ck = cv2.dilate(ink_ck, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        bad = sum(1 for px, py in gt["curve_px"]
                  if 0 <= px < img.shape[1] and 0 <= py < img.shape[0] and ink_ck[py, px] == 0)
        if bad > 3:
            print(f"[{idx}] WARNING: self-check: {bad} GT points not near ink")
            n_fail += 1

        img, applied = degrade(img, rng)
        gt["degradations"] = applied
        gt["image_size"] = [int(img.shape[1]), int(img.shape[0])]
        gt["line_style"] = cfg["ls"]
        gt["line_width"] = cfg["lw"]
        gt["legend"] = cfg["legend"]
        gt["grid"] = cfg["grid"]
        gt["spines"] = cfg["spines"]
        gt["color"] = cfg["color"]

        cv2.imwrite(stem + ".png", img)
        cv2.imwrite(stem + "_mask.png", mask_img)
        with open(stem + ".csv", "w", encoding="utf-8") as f:
            f.write("x,y\n")
            for tx, ty in zip(t, y):
                f.write(f"{tx:.8g},{ty:.8g}\n")
        with open(stem + "_meta.json", "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=1)
        with open(stem + "_labels.json", "w", encoding="utf-8") as f:
            json.dump(labels, f)
        print(f"[{idx}] ok  x={cfg['x_kind']:<6} y={cfg['y_kind']:<6} "
              f"fn={cfg['fn']:<8} dpi={cfg['dpi']} degrad={applied or 'none'}")

    print(f"done: {args.count - n_fail}/{args.count} generated, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
