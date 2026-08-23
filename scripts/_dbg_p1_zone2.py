import sys, os, json
sys.path.insert(0, 'src')
import numpy as np
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import _filter_mask_fragments, _embed_merge_masks
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.utils import read_image
from scipy.ndimage import binary_dilation

cfg = load_config()
segmenter = MultiUNetSegmenter('models/checkpoints/unet_multi_goi.pt', size=512)
stem = 'data/val_multi/img_0069'
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
print('emb shape:', None if emb is None else emb.shape)
ph = pw = 0
for m in masks:
    if m is not None:
        ph, pw = m.shape
        break
emb_c = emb[:, :ph, :pw] if emb is not None else None

# distance check between channel masks
from scipy.ndimage import distance_transform_edt
for c in range(len(masks)):
    if masks[c] is None: continue
    for j in range(c + 1, len(masks)):
        if masks[j] is None: continue
        d = distance_transform_edt(1 - masks[c].astype(np.uint8))[masks[j] > 0]
        mind = float(d.min()) if len(d) else float('inf')
        overlap = int(((masks[c] > 0) & (masks[j] > 0)).sum())
        dilated = binary_dilation(masks[j], iterations=4)
        zone = int(((masks[c] > 0) & dilated).sum())
        print(f'ch {c}-{j}: min_dist={mind:.1f}px overlap_px={overlap} zone_c_in_j4={zone}')

# centroid ok status
for c in range(len(masks)):
    m = masks[c]
    if m is None: continue
    conf = (m > 0) & (reg[c] > 0.6)
    print(f'ch {c}: mask_px={int(m.sum())} conf_px={int(conf.sum())} prob_max={float(reg[c].max()):.2f}')
