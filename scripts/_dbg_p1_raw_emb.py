import sys, os
sys.path.insert(0, 'src')
import numpy as np
import torch
import cv2
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
amax = np.argmax(reg, axis=0)
masks = []
for c in range(prob.shape[0]):
    region = reg[c]
    if float(region.max()) < thr:
        masks.append(None); continue
    mask01 = ((region > thr) & (amax == c)).astype(np.uint8)
    mask01 = _filter_mask_fragments(mask01, x1 - x0 + 1, y1 - y0 + 1)
    if int(mask01.sum()) < 16:
        masks.append(None); continue
    masks.append(mask01)

# RAW embedding at model resolution (no resize): replicate segmenter internals
h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
x = torch.from_numpy(small).float().unsqueeze(0).unsqueeze(0) / 255.0
with torch.no_grad():
    _, emb_raw = segmenter.model.forward_embed(x.to(segmenter.device))
emb_raw = torch.nn.functional.normalize(emb_raw, dim=1)[0].cpu().numpy()  # (E, 512, 512)
print('raw emb shape:', emb_raw.shape, 'norm range:', emb_raw.std())

# map plot bbox to 512-space
sx = 512.0 / w
sy = 512.0 / h
px0, py0, px1, py1 = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
reg_emb = emb_raw[:, py0:py1 + 1, px0:px1 + 1]  # (E, ph', pw') different scale than masks
print('reg_emb shape:', reg_emb.shape)

# centroids from confident pixels in RAW emb space (need to downsample masks to emb scale)
import skimage.transform
m0 = skimage.transform.resize(masks[0].astype(float), reg_emb.shape[1:], order=0) > 0.5
m1 = skimage.transform.resize(masks[1].astype(float), reg_emb.shape[1:], order=0) > 0.5
r0 = skimage.transform.resize(reg[0], reg_emb.shape[1:], order=1)
r1 = skimage.transform.resize(reg[1], reg_emb.shape[1:], order=1)
conf0 = m0 & (r0 > 0.6)
conf1 = m1 & (r1 > 0.6)
c0 = reg_emb[:, conf0].mean(axis=1); c0 /= np.linalg.norm(c0)
c1 = reg_emb[:, conf1].mean(axis=1); c1 /= np.linalg.norm(c1)
print('centroid cosine similarity:', float(c0 @ c1))
print('centroid norms:', float(np.linalg.norm(reg_emb[:, conf0].mean(axis=1))), float(np.linalg.norm(reg_emb[:, conf1].mean(axis=1))))

# approach zone in raw space
dil1 = binary_dilation(m1, iterations=2)
zone = m0 & dil1
print('raw zone px:', int(zone.sum()))
if int(zone.sum()) > 4:
    pz = reg_emb[:, zone]
    sc0 = np.asarray([float(c0 @ pz[:, i]) for i in range(pz.shape[1])])
    sc1 = np.asarray([float(c1 @ pz[:, i]) for i in range(pz.shape[1])])
    diff = sc0 - sc1
    print('raw diff: min=%.3f max=%.3f mean=%.3f' % (diff.min(), diff.max(), diff.mean()))
    print('raw frac |diff|>0.08:', float((np.abs(diff) > 0.08).mean()))
    print('sc0 mean=%.3f std=%.3f | sc1 mean=%.3f std=%.3f' % (sc0.mean(), sc0.std(), sc1.mean(), sc1.std()))
