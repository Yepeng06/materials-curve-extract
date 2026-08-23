import sys, os, json
sys.path.insert(0, 'src')
import numpy as np
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves_multi, _resolve_approach_zones, _embed_merge_masks, _filter_mask_fragments
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.utils import read_image

cfg = load_config()
segmenter = MultiUNetSegmenter('models/checkpoints/unet_multi_goi.pt', size=512)
stem = 'data/val_multi/img_0069'  # 3 crossing_jump failures
img = read_image(stem + '.png')
structure = detect_structure(img, cfg)
ocr = StubOCRBackend(stem + '_labels.json')
x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
xa, ya = build_axes(x_ticks, y_ticks,
                    x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
                    y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)))
prob = segmenter.prob_full(img)
x0, y0, x1, y1 = structure.plot_bbox
reg = prob[:, y0:y1 + 1, x0:x1 + 1]
thr = float(cfg.get('multi_mask_thr', 0.3))
independent = bool(cfg.get('multi_independent_mask', True))
min_area = int(cfg.get('multi_min_area', 450))
amax = np.argmax(reg, axis=0)
masks = []
for c in range(prob.shape[0]):
    region = reg[c]
    if float(region.max()) < thr:
        masks.append(None); continue
    if independent:
        mask01 = (region > thr).astype(np.uint8)
    else:
        mask01 = ((region > thr) & (amax == c)).astype(np.uint8)
    mask01 = _filter_mask_fragments(mask01, x1 - x0 + 1, y1 - y0 + 1)
    if int(mask01.sum()) < 16:
        masks.append(None); continue
    if min_area > 0 and int(mask01.sum()) < min_area:
        masks.append(None); continue
    masks.append(mask01)

emb = segmenter.embed_full(img)
masks_merged = _embed_merge_masks([m.copy() if m is not None else None for m in masks], reg, emb, min_area, x1 - x0 + 1, y1 - y0 + 1)
masks_resolved = _resolve_approach_zones([m.copy() if m is not None else None for m in masks_merged], reg, emb)

changed = 0
for c in range(len(masks)):
    if masks_resolved[c] is None or masks_merged[c] is None:
        continue
    d = int(np.abs(masks_resolved[c].astype(int) - masks_merged[c].astype(int)).sum())
    if d:
        print(f'channel {c}: {d} pixels changed by approach-zone resolution')
        changed += d
print('total changed pixels:', changed)
