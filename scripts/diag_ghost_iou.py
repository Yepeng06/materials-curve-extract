"""Check whether ghost channels overlap real channels (over-segmentation)."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
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

ghost_ious = []
examples = []
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
    masks = {}
    for c in range(prob.shape[0]):
        m = reg[c]
        if float(m.max()) < 0.3:
            continue
        mask01 = ((m > 0.3) & (amax == c)).astype(np.uint8)
        mask01 = _filter_mask_fragments(mask01, x1-x0+1, y1-y0+1)
        if int(mask01.sum()) < 16:
            continue
        masks[c] = mask01.astype(bool)
    for gc, gmask in masks.items():
        if gc in matched_chans:
            continue
        best = 0.0; best_rc = None
        for rc, rmask in masks.items():
            if rc == gc or rc not in matched_chans:
                continue
            inter = (gmask & rmask).sum()
            union = (gmask | rmask).sum()
            iou = inter / max(union, 1)
            if iou > best:
                best, best_rc = iou, rc
        ghost_ious.append(best)
        if len(examples) < 15:
            examples.append((name, gc, round(best, 3), best_rc, int(gmask.sum())))

ga = np.array(ghost_ious)
print(f"ghost channels: {len(ga)}, max-IoU-vs-real: min={ga.min():.3f} med={np.median(ga):.3f} p90={np.percentile(ga,90):.3f} max={ga.max():.3f}")
print(f"  with IoU>0.5 vs a real channel: {(ga>0.5).sum()} ({(ga>0.5).sum()/len(ga)*100:.0f}%)")
print(f"  with IoU>0.2: {(ga>0.2).sum()} ({(ga>0.2).sum()/len(ga)*100:.0f}%)")
for e in examples:
    print("  ", e)