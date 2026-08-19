"""Where is the model's probability peak relative to GT pixel center?
Samples the (K,H,W) prob map at GT curve pixels to measure the peak offset —
tells us whether the +1px bias is a learned model offset (training target)
or a centroid-algorithm artifact."""
from __future__ import annotations
import argparse, csv as csvlib, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import cv2
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/checkpoints/unet_multi_curve_512c.pt")
    ap.add_argument("--images", nargs="*", default=["data/val_multi/img_0101.png",
                    "data/val_multi/img_0021.png"])
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    seg = MultiUNetSegmenter(args.model, size=args.size)
    for img_path in args.images:
        stem = os.path.splitext(img_path)[0]
        img = read_image(img_path)
        h, w = img.shape[:2]
        prob = seg.prob_full(img)  # (K,H,W)
        # GT pixels via labels mapping (same as training target)
        cfg = None
        from mci.pipeline.extractor import load_config
        cfg = load_config()
        structure = detect_structure(img, cfg)
        ocr = StubOCRBackend(stem + "_labels.json")
        x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
            y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
        )
        with open(stem + "_curves.json", encoding="utf-8") as f:
            cj = json.load(f)
        gts = []
        for c in cj["curves"]:
            with open(os.path.join(os.path.dirname(stem), c["csv"]), encoding="utf-8") as f:
                rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
            gts.append(np.array([[float(r[0]), float(r[1])] for r in rows[1:]]))
        print(f"== {os.path.basename(img_path)}")
        for gi, g in enumerate(gts):
            gx = np.array([x_axis.value_to_pixel(v) for v, w in g])
            gy = np.array([y_axis.value_to_pixel(w) for v, w in g])
            offsets = []
            peaks = []
            for x, y in zip(gx[::5], gy[::5]):
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < w and 0 <= yi < h): continue
                win = prob[gi, max(0,yi-8):yi+9, max(0,xi-8):xi+9]
                if win.max() < 0.3: continue
                # peak position within window
                py, px = np.unravel_index(np.argmax(win), win.shape)
                offsets.append(py - 8)  # + = model peak below GT center
                peaks.append(win.max())
            if offsets:
                print(f"  GT{gi}: peak_offsets mean={np.mean(offsets):+.2f}px "
                      f"median={np.median(offsets):+.1f}px "
                      f"p90={np.percentile(np.abs(offsets),90):.1f}px "
                      f"mean_peak_prob={np.mean(peaks):.2f} n={len(offsets)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
