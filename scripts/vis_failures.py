"""Visualize prediction vs GT for specific failure images (Phase C diag)."""
from __future__ import annotations
import argparse, csv as csvlib, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import cv2, numpy as np
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
    ap.add_argument("--out-dir", default="data/eval_multi_diag_512c/vis")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--images", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config()
    segmenter = MultiUNetSegmenter(args.model, size=args.size)
    os.makedirs(args.out_dir, exist_ok=True)
    imgs = args.images
    if not imgs:
        imgs = ["data/val_multi/img_0015.png", "data/val_multi/img_0010.png",
                "data/val_multi/img_0067.png", "data/val_multi/img_0025.png"]
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
        curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
        # GT curves
        gts = []
        with open(stem + "_curves.json", encoding="utf-8") as f:
            cj = json.load(f)
        for c in cj["curves"]:
            with open(os.path.join(os.path.dirname(stem), c["csv"]), encoding="utf-8") as f:
                rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
            gts.append(np.array([[float(r[0]), float(r[1])] for r in rows[1:]]))
        rgb = img.copy()
        # draw plot bbox
        x0, y0, x1, y1 = structure.plot_bbox
        cv2.rectangle(rgb, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
        # GT in green
        for gi, g in enumerate(gts):
            px = [(x_axis.value_to_pixel(v), y_axis.value_to_pixel(v)) for v, w in g]
            arr = np.array([(int(a), int(b)) for a, b in px], dtype=np.int32)
            cv2.polylines(rgb, [arr], False, (0, 200, 0), 2)
        # pred in red/blue alternating
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for ci, c in enumerate(curves):
            arr = np.array([(int(a), int(b)) for a, b in c.pixel_points], dtype=np.int32)
            cv2.polylines(rgb, [arr], False, colors[ci % len(colors)], 2)
        out = os.path.join(args.out_dir, os.path.basename(stem) + "_vis.png")
        cv2.imwrite(out, rgb)
        print("wrote", out, "n_pred =", len(curves))
    return 0

if __name__ == "__main__":
    sys.exit(main())
