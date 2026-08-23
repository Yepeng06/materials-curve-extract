import sys, os
sys.path.insert(0, 'src')
import numpy as np
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import _filter_mask_fragments
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
min_area = int(cfg.get('multi_min_area', 450))
amax = np.argmax(reg, axis=0)
masks = []
for c in range(prob.shape[0]):
    region = reg[c]
    if float(region.max()) < thr:
        masks.append(None); continue
    mask01 = ((region > thr) & (amax == c)).astype(np.uint8)  # argmax version for clean curves
    mask01 = _filter_mask_fragments(mask01, x1 - x0 + 1, y1 - y0 + 1)
    if int(mask01.sum()) < 16:
        masks.append(None); continue
    masks.append(mask01)

emb = segmenter.embed_full(img)
ph = pw = 0
for m in masks:
    if m is not None:
        ph, pw = m.shape
        break
emb_c = emb[:, :ph, :pw]

# centroids from confident pixels
ctr, ok = [], []
for c in range(len(masks)):
    m = masks[c]
    if m is None:
        ctr.append(None); ok.append(False); continue
    conf = (m > 0) & (reg[c] > 0.6)
    if int(conf.sum()) < 8:
        ctr.append(None); ok.append(False); continue
    cv = emb_c[:, conf].mean(axis=1)
    n = float(np.linalg.norm(cv))
    ctr.append(cv / n if n > 1e-6 else None); ok.append(n > 1e-6)

# examine pair 0-1 approach zone
c, j = 0, 1
dilated_j = binary_dilation(masks[j], iterations=4)
zone = (masks[c] > 0) & dilated_j
px = emb_c[:, zone]
sc_c = np.asarray([float(ctr[c] @ px[:, i]) for i in range(px.shape[1])])
sc_j = np.asarray([float(ctr[j] @ px[:, i]) for i in range(px.shape[1])])
diff = sc_c - sc_j
print(f'pair {c}-{j}: zone_px={int(zone.sum())}')
print('diff: min=%.3f max=%.3f mean=%.3f' % (diff.min(), diff.max(), diff.mean()))
print('frac |diff|>0.08:', float((np.abs(diff) > 0.08).mean()))
print('frac diff>0.08:', float((diff > 0.08).mean()))
print('frac diff<-0.08:', float((diff < -0.08).mean()))
print('sc_c: mean=%.3f std=%.3f' % (sc_c.mean(), sc_c.std()))
print('sc_j: mean=%.3f std=%.3f' % (sc_j.mean(), sc_j.std()))
