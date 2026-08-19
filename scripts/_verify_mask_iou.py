"""Verify new-path masks coincide with GT curve pixels (via the SAME
axis mapping used at eval): mask-to-GT distance in px, per channel."""
import sys, os, glob, json, csv as csvlib
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
dists = []
for p in sorted(glob.glob("data/val_multi/*.png"))[:120]:
    if p.endswith("_mask.png"):
        continue
    stem = os.path.splitext(p)[0]
    with open(stem + "_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    img = read_image(p)
    h, w = img.shape[:2]
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
    except Exception:
        continue
    masks = T._meta_instance_masks(meta, labels, cj, os.path.dirname(stem),
                                   (w, h), axes=(x_axis, y_axis))
    for i, c in enumerate(cj["curves"][:6]):
        m = masks[i]
        if m.sum() == 0:
            continue
        csv_path = os.path.join(os.path.dirname(stem), c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        data = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        gx = x_axis.value_to_pixel(np.where(data[:, 0] > 0, data[:, 0], 1e-9))
        gy = y_axis.value_to_pixel(np.where(data[:, 1] > 0, data[:, 1], 1e-9))
        gpts = np.stack([gx, gy], 1)
        # sample GT pixels and check mask coverage within 2px
        ys, xs = np.nonzero(m)
        mask_pts = np.stack([xs.astype(float), ys.astype(float)], 1)
        # distance from each GT sample to nearest mask pixel
        step = max(1, len(gpts) // 200)
        hits = 0; n = 0
        for gp in gpts[::step]:
            n += 1
            d = np.min(np.abs(mask_pts - gp).sum(1))
            if d <= 2:
                hits += 1
        dists.append((os.path.basename(p), i, hits / max(n, 1), n))
ok = [d for d in dists if d[2] >= 0.95]
print(f"channels checked: {len(dists)}, coverage>=0.95: {len(ok)}")
low = [d for d in dists if d[2] < 0.95]
for d in low[:12]:
    print("  LOW:", d)
