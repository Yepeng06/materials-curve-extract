"""Area distribution of matched vs ghost channels across val_multi.

Uses the diag.json matching (pred_chan per GT) to label channels:
ghost = active channel that matched no GT.  Reports area quantiles for
real vs ghost channels to pick a safe suppression threshold.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# recompute per-channel stats quickly from diag + prob maps is slow; instead
# reuse the existing per-image pred counts and the ghost_diag output.
import glob, cv2
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.curve_extractor import _filter_mask_fragments
from mci.utils import read_image

cfg = load_config()
seg = MultiUNetSegmenter("models/checkpoints/unet_multi_curve_512c.pt", size=512)

with open("data/eval_multi_diag_512c/diag.json", encoding="utf-8") as f:
    diag = json.load(f)
rows = {r["image"]: r for r in diag["rows"]}

real_areas, ghost_areas = [], []
for p in sorted(glob.glob("data/val_multi/*.png")):
    if p.endswith("_mask.png"):
        continue
    stem = os.path.splitext(p)[0]
    name = os.path.basename(p)
    row = rows.get(name)
    if row is None or row.get("n_gt", -1) < 0:
        continue
    img = read_image(p)
    structure = detect_structure(img, cfg)
    x0, y0, x1, y1 = structure.plot_bbox
    prob = seg.prob_full(img)
    reg = prob[:, y0:y1+1, x0:x1+1]
    amax = np.argmax(reg, axis=0)
    matched_chans = {g["pred_chan"] for g in row["per_gt"] if g["pred_chan"] is not None}
    for c in range(prob.shape[0]):
        m = reg[c]
        if float(m.max()) < 0.3:
            continue
        mask01 = ((m > 0.3) & (amax == c)).astype(np.uint8)
        mask01 = _filter_mask_fragments(mask01, x1-x0+1, y1-y0+1)
        area = int(mask01.sum())
        if area < 16:
            continue
        (real_areas if c in matched_chans else ghost_areas).append(area)

ra = np.array(real_areas); ga = np.array(ghost_areas)
print(f"real channels: n={len(ra)} area min={ra.min()} p10={np.percentile(ra,10):.0f} "
      f"med={np.median(ra):.0f} max={ra.max()}")
if len(ga):
    print(f"ghost channels: n={len(ga)} area min={ga.min()} p50={np.median(ga):.0f} "
          f"p90={np.percentile(ga,90):.0f} max={ga.max()}")
    # threshold candidates
    for thr in (100, 200, 300, 400, 500, 800, 1000):
        dropped_real = int((ra < thr).sum())
        kept_ghost = int((ga >= thr).sum())
        print(f"  thr={thr}: drops {dropped_real}/{len(ra)} real ({dropped_real/len(ra)*100:.1f}%), "
              f"keeps {kept_ghost}/{len(ga)} ghost")
