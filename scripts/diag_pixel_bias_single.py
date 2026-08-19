"""Same pixel-bias check but with the SINGLE-curve model (which hits 100%
recall on val_single) — to separate model precision from multi-channel effects."""
from __future__ import annotations
import argparse, csv as csvlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import UNetSegmenter
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/checkpoints/unet_curve.pt")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config()
    seg = UNetSegmenter(args.model, size=args.size)
    imgs = args.images or ["data/val_multi/img_0101.png", "data/val_multi/img_0021.png",
                            "data/val_multi/img_0010.png"]
    for img_path in imgs:
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
        curves = extract_curves(img, structure, x_axis, y_axis, cfg, seg)
        pred_px = np.array([(px, py) for px, py in curves[0].pixel_points], dtype=np.float64)
        gts = []
        with open(stem + "_curves.json", encoding="utf-8") as f:
            cj = json.load(f)
        for c in cj["curves"]:
            with open(os.path.join(os.path.dirname(stem), c["csv"]), encoding="utf-8") as f:
                rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
            gts.append(np.array([[float(r[0]), float(r[1])] for r in rows[1:]]))
        print(f"== {os.path.basename(img_path)}: single-curve model")
        for gi, g in enumerate(gts):
            g_px = np.array([(x_axis.value_to_pixel(v), y_axis.value_to_pixel(w)) for v, w in g])
            g_px = g_px[np.argsort(g_px[:, 0])]
            p = pred_px[np.argsort(pred_px[:, 0])]
            x_lo = max(g_px[0,0], p[0,0]); x_hi = min(g_px[-1,0], p[-1,0])
            inside = (p[:,0] >= x_lo) & (p[:,0] <= x_hi)
            if inside.sum() < 3: continue
            gy = np.interp(p[inside,0], g_px[:,0], g_px[:,1])
            dy = p[inside,1] - gy
            print(f"  vs GT{gi}: px_bias={np.mean(dy):+.2f}px px_rmse={np.sqrt(np.mean(dy**2)):.2f}px max={np.max(np.abs(dy)):.1f}px n={inside.sum()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
