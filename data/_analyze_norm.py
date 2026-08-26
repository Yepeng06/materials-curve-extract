import json, os
import numpy as np

base = r"F:\CODE\New\baseline\data"
with open(os.path.join(base, "eval_multi_diag_chain", "diag.json"), encoding="utf-8") as f:
    diag = json.load(f)

SRC_DIRS = [os.path.join(base, "val_multi"), os.path.join(base, "val_single")]

def meta_y_range(name):
    stem = os.path.splitext(name)[0]
    for d in SRC_DIRS:
        mp = os.path.join(d, stem + "_meta.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta = json.load(f)
            yr = meta.get("y_range")
            if yr and yr[1] > yr[0]:
                return float(yr[1] - yr[0])
    return None

rows = diag["rows"]
total = 0; rec_span = 0; rec_fs = 0
flip_span_to_fs = 0; flip_fs_to_span = 0; no_meta = 0
ratios = []
for r in rows:
    if r.get("n_gt", -1) <= 0:
        continue
    yr = meta_y_range(r["image"])
    if yr is None:
        no_meta += 1
    for g in r.get("per_gt", []):
        rel_span = g["rel"]; y_span = g["y_span"]
        total += 1
        if rel_span == float("inf") or y_span <= 0 or yr is None:
            continue
        rmse = rel_span * y_span
        rel_fs = rmse / yr
        ratios.append(rel_fs / rel_span)
        ok_span = rel_span <= 0.01; ok_fs = rel_fs <= 0.01
        rec_span += int(ok_span); rec_fs += int(ok_fs)
        if ok_span and not ok_fs: flip_span_to_fs += 1
        if ok_fs and not ok_span: flip_fs_to_span += 1

print(f"total GT curves: {total}  (meta missing: {no_meta})")
print(f"recalled by span-norm (current): {rec_span} ({rec_span/total:.4f})")
print(f"recalled by fullscale-norm (acceptance): {rec_fs} ({rec_fs/total:.4f})")
print(f"flips span-OK->fullscale-FAIL: {flip_span_to_fs}, fullscale-OK->span-FAIL: {flip_fs_to_span}")
a = np.array(ratios)
print(f"ratio fs/span: median {np.median(a):.3f}  p90 {np.percentile(a,90):.3f}  max {np.max(a):.3f}")
