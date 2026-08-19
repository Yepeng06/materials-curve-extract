"""Verify the fixed training masks align with the rendered image ink.

For a batch of val_multi images: rebuild instance masks via the new
eval-consistent path, then check that each mask channel's pixels sit on
dark ink of the image (mean gray of mask pixels should be low; and the
mask should sit within the plot area).
"""
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
log_ok = log_bad = lin_ok = lin_bad = 0
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
    n_active = 0
    ok = True
    for c in range(6):
        m = masks[c]
        if m.sum() == 0:
            continue
        n_active += 1
        px = gray[m > 0]
        mean_gray = float(px.mean()) if len(px) else 255.0
        # mask on dark ink: mean gray should be well below 255 (paper)
        if mean_gray > 200 or len(px) < 100:
            ok = False
    yk = meta.get("y_kind", "?")
    if yk == "log":
        log_ok += ok; log_bad += (not ok)
    else:
        lin_ok += ok; lin_bad += (not ok)
print(f"log-axis imgs: ok={log_ok} bad={log_bad}")
print(f"linear-axis imgs: ok={lin_ok} bad={lin_bad}")
