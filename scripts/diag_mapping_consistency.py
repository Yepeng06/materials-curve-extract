"""Check training-target mapping (_fit_axis_from_labels in
train_segmentation_multi.py) vs evaluation mapping (build_axes via
read_ticks): do they agree on the same labels.json? A systematic
difference would mean the model learns a target offset from the
evaluation basis (explains the +1px learned bias)."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import cv2
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.extractor import load_config
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

# replicate train_segmentation_multi._fit_axis_from_labels
def fit_axis_from_labels(labels, axis, img_h):
    from mci.utils import parse_number_text
    pts = []
    for it in labels:
        box = np.asarray(it["box"], dtype=float)
        c = box.mean(axis=0)
        v = parse_number_text(str(it["text"]))
        if v is None:
            continue
        if axis == "x" and c[1] > img_h - 80:
            pts.append((float(c[0]), v))
        elif axis == "y" and c[0] < 100 and c[1] < img_h - 85:
            pts.append((float(c[1]), v))
    if len(pts) < 2:
        return None
    pts.sort()
    p = np.array([q[0] for q in pts])
    v = np.array([q[1] for q in pts])
    if (v > 0).all() and (float(v.max()) / float(v.min()) > 100):
        a, b = np.polyfit(p, np.log10(v), 1)
        return ("log", float(a), float(b))
    a, b = np.polyfit(p, v, 1)
    return ("linear", float(a), float(b))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=["data/val_multi/img_0101.png",
                    "data/val_multi/img_0021.png", "data/val_multi/img_0010.png",
                    "data/val_multi/img_0067.png", "data/val_multi/img_0025.png"])
    args = ap.parse_args()
    cfg = load_config()
    for img_path in args.images:
        stem = os.path.splitext(img_path)[0]
        img = read_image(img_path)
        h, w = img.shape[:2]
        with open(stem + "_labels.json", encoding="utf-8") as f:
            labels = json.load(f)
        ft = fit_axis_from_labels(labels, "y", h)
        structure = detect_structure(img, cfg)
        ocr = StubOCRBackend(stem + "_labels.json")
        x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
            y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
        )
        # compare: for sample y values, pixel per train-fit vs eval-axis
        print(f"== {os.path.basename(img_path)}")
        print(f"   train-fit: {ft}")
        print(f"   eval-axis: kind={y_axis.kind} slope={y_axis.slope:.6f} intercept={y_axis.intercept:.5f} "
              f"sign={y_axis.sign} pmin={y_axis.pmin:.1f} pmax={y_axis.pmax:.1f}")
        vals = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        if ft is not None:
            k, a, b = ft
            for v in vals:
                # train pixel
                if k == "log":
                    p_t = (np.log10(v) - b) / a
                else:
                    p_t = (v - b) / a
                p_e = y_axis.value_to_pixel(v)
                print(f"   v={v:5g}: train_px={p_t:8.2f} eval_px={p_e:8.2f} diff={p_t-p_e:+.2f}px")
    return 0

if __name__ == "__main__":
    sys.exit(main())
