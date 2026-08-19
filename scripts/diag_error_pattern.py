"""Deep error-pattern analysis: where do failed curves diverge from GT?

For each failed GT curve (rel > 1%), compute:
  - x-range overlap fraction (does the prediction cover the GT span?)
  - signed bias (mean pred - GT in value space, in % of y-span)
  - error concentration: which x-decile of the GT span carries the error
  - chain length / number of predicted points
This distinguishes: (a) missing tail, (b) whole-curve bias, (c) local spikes
(crossing/attribution switches), (d) wrong-curve match.
"""
from __future__ import annotations
import argparse, csv as csvlib, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
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

def match_pairs(gts, preds):
    n_gt, n_pred = len(gts), len(preds)
    costs = [[curve_rmse(g[1], p) for p in preds] for g in gts]
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
    return pairs

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/checkpoints/unet_multi_curve_512c.pt")
    ap.add_argument("--out", default="data/eval_multi_diag_512c/error_pattern.json")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    segmenter = MultiUNetSegmenter(args.model, size=args.size)

    images = sorted(glob.glob("data/val_multi/*.png"))
    images = [p for p in images if not os.path.basename(p).endswith("_mask.png")]
    if args.limit: images = images[:args.limit]

    out_rows = []
    for img_path in images:
        stem = os.path.splitext(img_path)[0]
        img = read_image(img_path)
        structure = detect_structure(img, cfg)
        ocr = StubOCRBackend(stem + "_labels.json")
        x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
            y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
        )
        try:
            curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
        except Exception:
            continue
        gts = load_gt_curves(stem)
        preds = [np.asarray(c.points, dtype=np.float64) for c in curves]
        pairs = match_pairs(gts, preds)
        for i, (label, g) in enumerate(gts):
            g = g[np.argsort(g[:,0])]
            y_span = float(g[:,1].max()-g[:,1].min())
            m = next((rmse for gi,_,rmse in pairs if gi==i), float("inf"))
            rel = m/y_span if y_span>0 else float("inf")
            if rel <= 0.01: continue  # only failures
            pj = next((j for gi,j,_ in pairs if gi==i), None)
            detail = {"image": os.path.basename(img_path), "curve": label,
                      "rel": round(rel,4), "y_span": round(y_span,5)}
            if pj is None:
                detail["mode"] = "no_match"
                out_rows.append(detail); continue
            p = preds[pj][np.argsort(preds[pj][:,0])]
            # overlap
            x_lo = max(float(g[0,0]), float(p[0,0])); x_hi = min(float(g[-1,0]), float(p[-1,0]))
            g_span = float(g[-1,0]-g[0,0])
            overlap = (x_hi-x_lo)/g_span if g_span>0 else 0
            detail["x_overlap"] = round(overlap,3)
            detail["gt_x_range"] = [round(float(g[0,0]),3), round(float(g[-1,0]),3)]
            detail["pred_x_range"] = [round(float(p[0,0]),3), round(float(p[-1,0]),3)]
            detail["n_pred_pts"] = len(p)
            inside = (p[:,0] >= x_lo) & (p[:,0] <= x_hi)
            if inside.sum() < 3:
                detail["mode"] = "tiny_overlap"; out_rows.append(detail); continue
            gy = np.interp(p[inside,0], g[:,0], g[:,1])
            err = p[inside,1]-gy
            detail["bias_pct"] = round(float(np.mean(err)/y_span*100),3)   # signed bias
            detail["rmse_pct"] = round(float(np.sqrt(np.mean(err**2))/y_span*100),3)
            detail["max_abs_err_pct"] = round(float(np.max(np.abs(err))/y_span*100),3)
            # per-decile |err| concentration
            dec = np.digitize(p[inside,0], np.quantile(g[:,0], np.linspace(0,1,11)))
            errs = [float(np.mean(np.abs(err[dec==d]))/y_span*100) for d in range(1,11)]
            detail["decile_err_pct"] = [round(e,2) for e in errs]
            # mode classification
            if overlap < 0.6:
                detail["mode"] = "partial_trace"
            elif detail["max_abs_err_pct"] > 5 and max(detail["decile_err_pct"]) > 2*float(np.mean(detail["decile_err_pct"])) + 1:
                detail["mode"] = "local_spike"
            elif abs(detail["bias_pct"]) > 0.8*detail["rmse_pct"]:
                detail["mode"] = "systematic_bias"
            else:
                detail["mode"] = "mixed"
            out_rows.append(detail)
    from collections import Counter
    modes = Counter(r["mode"] for r in out_rows)
    print("total failed curves:", len(out_rows))
    print("modes:", dict(modes))
    from collections import defaultdict
    m = defaultdict(list)
    for r in out_rows: m[r["mode"]].append(r)
    for mode, rs in m.items():
        print(f"\n== {mode} ({len(rs)}) ==")
        for r in sorted(rs, key=lambda r: -r["rel"])[:12]:
            print(" ", r["image"], r["curve"], "rel=", r["rel"], "bias=", r.get("bias_pct"),
                  "rmse%=", r.get("rmse_pct"), "max=", r.get("max_abs_err_pct"),
                  "overlap=", r.get("x_overlap"), "pts=", r.get("n_pred_pts"))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"modes": dict(modes), "rows": out_rows}, f, indent=1, ensure_ascii=False)
    return 0

if __name__ == "__main__":
    sys.exit(main())
