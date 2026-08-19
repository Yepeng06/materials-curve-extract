"""Experiment: argmax-exclusive vs independent-threshold masks.
Checks whether removing the argmax exclusion lets both channels keep
pixels at crossings (LineFormer-style multi-label), enabling better
tracing through junctions, or whether it just fuses nearby curves."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.utils import read_image

cfg = load_config()
seg = MultiUNetSegmenter("models/checkpoints/unet_multi_curve_512c.pt", size=512)
thr = 0.3
amax_areas = []
indep_areas = []
for p in sorted(glob.glob("data/val_multi/*.png"))[:60]:
    if p.endswith("_mask.png"):
        continue
    stem = os.path.splitext(p)[0]
    img = read_image(p)
    structure = detect_structure(img, cfg)
    x0, y0, x1, y1 = structure.plot_bbox
    prob = seg.prob_full(img)
    reg = prob[:, y0:y1+1, x0:x1+1]
    amax = np.argmax(reg, axis=0)
    over = (reg > thr).sum(axis=0)
    n_overlap_px = int((over >= 2).sum())
    n_plot = int((over > 0).sum())
    excl_area = 0
    for c in range(reg.shape[0]):
        excl_area += int(((reg[c] > thr) & (amax == c)).sum())
    indep_area = int((reg > thr).sum())
    amax_areas.append(excl_area)
    indep_areas.append(indep_area)
    if n_plot and n_overlap_px / n_plot > 0.02:
        print(os.path.basename(p), "overlap_px=", n_overlap_px, "/", n_plot,
              "(%.1f%%)" % (100 * n_overlap_px / n_plot),
              "excl=", excl_area, "indep=", indep_area)
a = np.array(amax_areas); b = np.array(indep_areas)
print("mean excl area:", round(a.mean()), "mean indep area:", round(b.mean()),
      "mean extra px (indep-excl):", round((b - a).mean()))