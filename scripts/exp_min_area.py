"""Quick experiment: does a min-area filter on extracted channels help 6a/6b?"""
import json, os, sys, glob, csv as csvlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.curve_extractor import extract_curves_multi
from mci.pipeline.extractor import load_config
from mci.pipeline.segmenter import MultiUNetSegmenter
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image

def load_gt_curves(stem):
    out = []
    with open(stem + "_curves.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data["curves"]:
        with open(os.path.join(os.path.dirname(stem), c["csv"]), encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        pts = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        out.append((c.get("label") or c["curve_id"], pts))
    return out

def curve_rmse(gt, pred):
    g = gt[np.argsort(gt[:, 0])]; p = pred[np.argsort(pred[:, 0])]
    x_lo = max(float(g[0,0]), float(p[0,0])); x_hi = min(float(g[-1,0]), float(p[-1,0]))
    inside = (p[:,0] >= x_lo) & (p[:,0] <= x_hi)
    if inside.sum() < 2: return float("inf")
    gy = np.interp(p[inside,0], g[:,0], g[:,1])
    return float(np.sqrt(np.mean((p[inside,1]-gy)**2)))

cfg = load_config()
seg = MultiUNetSegmenter("models/checkpoints/unet_multi_curve_512c.pt", size=512)

for min_area in (0, 100, 200, 400):
    recalled = 0; n_gt_tot = 0; n_pred_tot = 0; cok = 0; nimg = 0
    for p in sorted(glob.glob("data/val_multi/*.png"))[:120]:
        if p.endswith("_mask.png"): continue
        stem = os.path.splitext(p)[0]
        img = read_image(p)
        structure = detect_structure(img, cfg)
        ocr = StubOCRBackend(stem + "_labels.json")
        x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
            y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
        )
        cfg2 = dict(cfg); cfg2["multi_min_area"] = min_area
        curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg2, seg)
        gts = load_gt_curves(stem)
        preds = [np.asarray(c.points, dtype=np.float64) for c in curves]
        n_gt = len(gts); n_pred = len(preds)
        costs = [[curve_rmse(g[1], p_) for p_ in preds] for g in gts]
        mg = [False]*n_gt; mp = [False]*n_pred; pairs = []
        while True:
            best = None
            for i in range(n_gt):
                if mg[i]: continue
                for j in range(n_pred):
                    if mp[j]: continue
                    c = costs[i][j]
                    if c == float("inf"): continue
                    if best is None or c < best[0]: best = (c,i,j)
            if best is None: break
            _,i,j = best; mg[i]=True; mp[j]=True; pairs.append((i,j,best[0]))
        rels = []
        for i,(label,g) in enumerate(gts):
            y_span = float(g[:,1].max()-g[:,1].min())
            m = next((rmse for gi,_,rmse in pairs if gi==i), float("inf"))
            rels.append(m/y_span if y_span>0 else float("inf"))
        recalled += sum(1 for r in rels if r <= 0.01)
        n_gt_tot += n_gt; n_pred_tot += n_pred
        cok += int(n_gt == n_pred); nimg += 1
    r6a = recalled/n_gt_tot; r6b = recalled/max(n_gt_tot, n_pred_tot)
    print(f"min_area={min_area}: 6a={r6a:.4f} 6b={r6b:.4f} n_pred={n_pred_tot} curve-count-ok={cok}/{nimg}")