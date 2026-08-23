import sys, os
sys.path.insert(0, 'src')
import numpy as np
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.tick_reader import PaddleOCRBackend, read_ticks, _ocr_strips, _ocr_strip_scaled, _classify_labels, _is_tick_label, _strip_crop
from mci.pipeline.extractor import load_config
from mci.utils import read_image

cfg = load_config()
ocr = PaddleOCRBackend()
# from R1e: PMC3556318_1471-2458-13-41-1.jpg read 10-12 x labels in my strip test but pipeline failed
path = 'data/real_diag/pmc_oa/raw/PMC3556318/1471-2458-13-41-1.jpg'
if not os.path.exists(path):
    # find it
    import glob
    cands = glob.glob('data/real_diag/**/1471-2458-13-41-1.jpg', recursive=True)
    print('candidates:', cands)
    path = cands[0] if cands else None
if not path:
    sys.exit(1)
img = read_image(path)
if img.ndim == 3 and img.shape[2] == 4:
    a = img[:, :, 3] / 255.0
    img = (img[:, :, :3] * a[..., None] + 255 * (1 - a[..., None])).astype(np.uint8)
print('image:', path, img.shape)
try:
    structure = detect_structure(img, cfg)
    print('plot_bbox:', structure.plot_bbox)
    print('x_axis_pixel:', structure.x_axis_pixel, 'y_axis_pixel:', structure.y_axis_pixel)
    print('x_ticks_px:', structure.x_ticks_px)
    print('y_ticks_px:', structure.y_ticks_px)
except Exception as e:
    print('structure failed:', type(e).__name__, e)
    sys.exit(0)

# 1. pipeline strip OCR at 2x
boxes2 = _ocr_strips(img, structure, ocr, cfg)
print('\n2x strip boxes (tick-filtered):')
for b in boxes2:
    if _is_tick_label(b, 0.55):
        print('  ', repr(b.text), b.score, 'center=(%.0f,%.0f)' % (b.center[0], b.center[1]))
# 2. my experiment strip (bottom band)
x0, y0, x1, y1 = structure.plot_bbox
h, w = img.shape[:2]
my_strip = img[max(0, y1): min(h, y1 + 90), max(0, x0 - 40): min(w, x1 + 40)]
up = cv2_resize = None
import cv2
up = cv2.resize(my_strip, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
print('\nmy 2x bottom-band boxes:')
for b in ocr.read_text_boxes(up):
    print('  ', repr(b.text), round(b.score, 2))
# 3. pipeline x strip crop
crop, ox, oy = _strip_crop(img, structure, 'x')
print('\npipeline x strip: shape', crop.shape, 'offset', ox, oy)
up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
print('pipeline x strip 2x boxes:')
for b in ocr.read_text_boxes(up):
    print('  ', repr(b.text), round(b.score, 2))
