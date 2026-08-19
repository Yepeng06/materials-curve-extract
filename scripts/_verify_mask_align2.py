"""Detail the mask-alignment failures: mean gray per active channel,
plus whether the whole mask sits inside the plot bbox."""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import numpy as np
import cv2
import train_segmentation_multi as T
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.extractor import load_config
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

cfg = load_config()
for p in sorted(glob.glob("data/val_multi/*.png"))[:120]:
    if p.endswith("_mask.png"):
        continue
    stem = os.path.splitext(p)[0]
    with open(stem + "_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    img = read_image(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    with open(stem + "_labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    with open(stem + "_curves.json", encoding="utf-8") as f:
        cj = json.load(f)
    try:
        structure = detect_structure(img, cfg)
        ocr = StubOCRBackend(stem + "_labels.json")
        x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
            y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
        )
    except Exception as e:
        continue
    masks = T._meta_instance_masks(meta, labels, cj, os.path.dirname(stem),
                                   (w, h), axes=(x_axis, y_axis))
    bad = []
    for c in range(6):
        m = masks[c]
        if m.sum() == 0:
            continue
        px = gray[m > 0]
        mg = float(px.mean()) if len(px) else 255.0
        ys, xs = np.nonzero(m)
        in_plot = (xs.min() >= structure.plot_bbox[0] - 5 and xs.max() <= structure.plot_bbox[2] + 5 and
                   ys.min() >= structure.plot_bbox[1] - 5 and ys.max() <= structure.plot_bbox[3] + 5)
        bad.append((c, round(mg, 1), int(m.sum()), in_plot))
    if any(b[1] > 200 or not b[3] for b in bad):
        print(os.path.basename(p), "y_kind=", meta.get("y_kind"),
              "plot_bbox=", structure.plot_bbox, "bad=", bad)
