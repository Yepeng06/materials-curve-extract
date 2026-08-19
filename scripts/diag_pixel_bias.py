"""Check whether systematic_bias failures come from pixel-level model offset
or from the value mapping: compare predicted pixel y vs GT pixel y directly."""
from __future__ import annotations
import argparse, csv as csvlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves_multi
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/checkpoints/unet_multi_curve_512c.pt")
    ap.add_argument("--images", nargs="*", default=["data/val_multi/img_0021.png",
                    "data/val_multi/img_0101.png", "data/val_multi/img_0010.png"])
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config()
    segmenter = MultiUNetSegmenter(args.model, size=args.size)
    for img_path in args.images:
        stem = os.path.splitext(img_path)[0]
        img = read_image(img_path)
        structure = detect_structure(img, cfg)
        ocr = StubOCRBackend(stem + "_labels.json")
        x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
            y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
        )
        curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
        # GT data points -> pixels via the same axis
        gts = []
        with open(stem + "_curves.json", encoding="utf-8") as f:
            cj = json.load(f)
        for c in cj["curves"]:
            with open(os.path.join(os.path.dirname(stem), c["csv"]), encoding="utf-8") as f:
                rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
            gts.append(np.array([[float(r[0]), float(r[1])] for r in rows[1:]]))
        print(f"== {os.path.basename(img_path)}: plot_bbox={structure.plot_bbox} "
              f"y_axis: kind={y_axis.kind} slope={y_axis.slope:.6f} int={y_axis.intercept:.4f} "
              f"pmin={y_axis.pmin:.1f} pmax={y_axis.pmax:.1f} vmin={y_axis.vmin:.4g} vmax={y_axis.vmax:.4g}")
        for ci, c in enumerate(curves):
            pred_px = np.array([(px, py) for px, py in c.pixel_points], dtype=np.float64)
            # per-pred-pixel nearest GT in x, compare y in PIXEL space
            for gi, g in enumerate(gts):
                g_px = np.array([(x_axis.value_to_pixel(v), y_axis.value_to_pixel(w)) for v, w in g])
                g_px = g_px[np.argsort(g_px[:, 0])]
                p = pred_px[np.argsort(pred_px[:, 0])]
                x_lo = max(g_px[0,0], p[0,0]); x_hi = min(g_px[-1,0], p[-1,0])
                inside = (p[:,0] >= x_lo) & (p[:,0] <= x_hi)
                if inside.sum() < 3: continue
                gy = np.interp(p[inside,0], g_px[:,0], g_px[:,1])
                dy = p[inside,1] - gy
                print(f"  chan{ci} vs GT{gi}: px_bias={np.mean(dy):+.2f}px "
                      f"px_rmse={np.sqrt(np.mean(dy**2)):.2f}px "
                      f"max={np.max(np.abs(dy)):.1f}px n={inside.sum()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
