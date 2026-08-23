"""S2-multi: rerun the Phase-C multi-curve diagnostic on 2x-upsampled copies
of the images that contain 1-2% rel-error curves.

Reproduces scripts/eval_multi_diag.py exactly (same GT loading, same
greedy matching, same bucket definition, same segmenter/config) but on the
2x Lanczos-upsampled set, and compares per-(image, GT-index) buckets with
the baseline data/eval_multi_diag_goi/diag.json.

Usage:
  python scripts/diag_subpixel_s2_multi.py \
      --diag data/eval_multi_diag_goi/diag.json \
      --model models/checkpoints/unet_multi_goi.pt \
      --size 512 \
      --src-dirs data/val_multi,data/val_single \
      --out-dir data/experiments_s1s2
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from mci.pipeline.chart_structure import detect_structure  # noqa: E402
from mci.pipeline.coordinate_mapper import build_axes  # noqa: E402
from mci.pipeline.curve_extractor import extract_curves_multi  # noqa: E402
from mci.pipeline.extractor import load_config  # noqa: E402
from mci.pipeline.segmenter import MultiUNetSegmenter  # noqa: E402
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks  # noqa: E402
from mci.utils import read_image  # noqa: E402

import diag_subpixel_s2_prep as prep  # noqa: E402  (same dir)


def load_gt_curves(stem: str) -> list:
    out = []
    with open(stem + "_curves.json", encoding="utf-8") as f:
        data = json.load(f)
    for c in data["curves"]:
        csv_path = os.path.join(os.path.dirname(stem), c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        pts = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        out.append((c.get("label") or c["curve_id"], pts))
    return out


def curve_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    g = gt[np.argsort(gt[:, 0])]
    p = pred[np.argsort(pred[:, 0])]
    x_lo = max(float(g[0, 0]), float(p[0, 0]))
    x_hi = min(float(g[-1, 0]), float(p[-1, 0]))
    inside = (p[:, 0] >= x_lo) & (p[:, 0] <= x_hi)
    if inside.sum() < 2:
        return float("inf")
    gy = np.interp(p[inside, 0], g[:, 0], g[:, 1])
    return float(np.sqrt(np.mean((p[inside, 1] - gy) ** 2)))


def bucket(rel: float) -> str:
    if rel <= 0.01:
        return "ok"
    if rel <= 0.02:
        return "e1_2"
    if rel <= 0.05:
        return "e2_5"
    return "e5p"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag", default="data/eval_multi_diag_goi/diag.json")
    ap.add_argument("--model", default="models/checkpoints/unet_multi_goi.pt")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--src-dirs", default="data/val_multi,data/val_single")
    ap.add_argument("--out-dir", default="data/experiments_s1s2")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.diag, encoding="utf-8") as f:
        diag = json.load(f)
    base_rows = diag["rows"]

    # rebuild the exact input order the baseline run used
    images = []
    for d in args.src_dirs.split(","):
        d = d.strip()
        if not os.path.isdir(d):
            continue
        images += sorted(glob.glob(os.path.join(d, "*.png")))
    images = [p for p in images if not os.path.basename(p).endswith("_mask.png")]
    assert len(images) == len(base_rows), (len(images), len(base_rows))
    for p, r in zip(images, base_rows):
        assert os.path.basename(p) == r["image"], (p, r["image"])

    # select images with any 1-2% curve
    sel_idx = [i for i, r in enumerate(base_rows)
               if any(0.01 < g["rel"] <= 0.02 for g in r["per_gt"])]
    if args.limit:
        sel_idx = sel_idx[: args.limit]
    print(f"[s2multi] selected {len(sel_idx)} images with 1-2% curves")

    # ---- baseline buckets on the selected set (per image, per GT index) ----
    base_map = {}  # (image, gt_idx) -> (rel, bucket)
    for i in sel_idx:
        r = base_rows[i]
        for gi, g in enumerate(r["per_gt"]):
            base_map[(r["image"], gi)] = (g["rel"], bucket(g["rel"]))

    # ---- prep 2x copies ----
    x2_dir = os.path.join(args.out_dir, "multi_x2")
    os.makedirs(x2_dir, exist_ok=True)
    sel_stems = [os.path.splitext(images[i])[0] for i in sel_idx]
    select_file = os.path.join(args.out_dir, "multi_select.txt")
    with open(select_file, "w", encoding="utf-8") as f:
        for s in sel_stems:
            f.write(os.path.basename(s) + "\n")
    for s in sel_stems:
        _prep_one(s, x2_dir)

    # ---- run the multi pipeline on x2 copies ----
    cfg = load_config()
    segmenter = MultiUNetSegmenter(args.model, size=args.size)
    rows_x2 = []
    for s in sel_stems:
        base = os.path.basename(s)
        img_path = os.path.join(x2_dir, base + ".png")
        stem = os.path.join(x2_dir, base)
        name = base + ".png"
        row = {"image": name}
        try:
            img = read_image(img_path)
            structure = detect_structure(img, cfg)
            ocr = StubOCRBackend(stem + "_labels.json")
            x_ticks, y_ticks = read_ticks(img, structure, ocr, cfg)
            x_axis, y_axis = build_axes(
                x_ticks, y_ticks,
                x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
                y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
            )
            curves = extract_curves_multi(img, structure, x_axis, y_axis, cfg, segmenter)
            gts = load_gt_curves(stem)
            n_gt = len(gts)
            n_pred = len(curves)
            preds = [np.asarray(c.points, dtype=np.float64) for c in curves]
            costs = [[curve_rmse(g[1], p) for p in preds] for g in gts]
            matched_gt = [False] * n_gt
            matched_pred = [False] * n_pred
            pairs = []
            while True:
                best = None
                for i in range(n_gt):
                    if matched_gt[i]:
                        continue
                    for j in range(n_pred):
                        if matched_pred[j]:
                            continue
                        c = costs[i][j]
                        if c == float("inf"):
                            continue
                        if best is None or c < best[0]:
                            best = (c, i, j)
                if best is None:
                    break
                _, i, j = best
                matched_gt[i] = True
                matched_pred[j] = True
                pairs.append((i, j, best[0]))
            per_gt = []
            for i, (label, g) in enumerate(gts):
                y_span = float(g[:, 1].max() - g[:, 1].min())
                m = next((rmse for gi, _, rmse in pairs if gi == i), float("inf"))
                rel = m / y_span if y_span > 0 else float("inf")
                per_gt.append({"label": label, "rel": round(rel, 4), "y_span": round(y_span, 5)})
            row["per_gt"] = per_gt
            row["status"] = "ok"
            print(f"[OK ] {name}: gt={n_gt} pred={n_pred} per_gt={[(g['rel'], bucket(g['rel'])) for g in per_gt]}")
        except Exception as e:
            row["per_gt"] = []
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[ERR] {name}: {row['error']}")
        rows_x2.append(row)

    # ---- compare per (image, gt index) ----
    x2_map = {}
    for r in rows_x2:
        for gi, g in enumerate(r["per_gt"]):
            x2_map[(r["image"], gi)] = (g["rel"], bucket(g["rel"]))

    comp = {"unchanged_ok": 0, "ok_to_e1_2": 0, "e1_2_to_ok": 0,
            "e1_2_to_e2_5": 0, "e2_5_to_e1_2": 0, "e2_5_to_ok": 0,
            "to_e5p": 0, "from_e5p": 0, "worse_any": 0, "better_any": 0,
            "n_curves": 0, "detail": []}
    b_before = {"ok": 0, "e1_2": 0, "e2_5": 0, "e5p": 0, "inf": 0}
    b_after = {"ok": 0, "e1_2": 0, "e2_5": 0, "e5p": 0, "inf": 0}
    for key, (rel0, b0) in base_map.items():
        comp["n_curves"] += 1
        b_before[b0] += 1
        if key in x2_map:
            rel1, b1 = x2_map[key]
            b_after[b1] += 1
        else:
            b_after["inf"] += 1
            b1 = "inf"
        rank = {"ok": 0, "e1_2": 1, "e2_5": 2, "e5p": 3, "inf": 4}
        comp["detail"].append({"image": key[0], "gt_idx": key[1],
                               "before_rel": rel0, "before": b0,
                               "after_rel": x2_map.get(key, (None, "inf"))[0],
                               "after": b1})
        if rank[b1] < rank[b0]:
            comp["better_any"] += 1
        elif rank[b1] > rank[b0]:
            comp["worse_any"] += 1
        if b0 == "ok" and b1 == "e1_2":
            comp["ok_to_e1_2"] += 1
        if b0 == "e1_2" and b1 == "ok":
            comp["e1_2_to_ok"] += 1
        if b0 == "e1_2" and b1 == "e2_5":
            comp["e1_2_to_e2_5"] += 1
        if b0 == "e2_5" and b1 == "e1_2":
            comp["e2_5_to_e1_2"] += 1
        if b0 == "e2_5" and b1 == "ok":
            comp["e2_5_to_ok"] += 1
        if b0 in ("ok", "e1_2", "e2_5") and b1 == "e5p":
            comp["to_e5p"] += 1
        if b0 == "e5p" and b1 in ("ok", "e1_2", "e2_5"):
            comp["from_e5p"] += 1

    summary = {
        "model": args.model, "size": args.size,
        "n_images_selected": len(sel_idx),
        "n_curves": comp["n_curves"],
        "buckets_before": b_before,
        "buckets_after": b_after,
        "recall_before": b_before["ok"] / comp["n_curves"] if comp["n_curves"] else 0,
        "recall_after": b_after["ok"] / comp["n_curves"] if comp["n_curves"] else 0,
        "transitions": {k: v for k, v in comp.items() if k != "detail"},
        "rows_x2": rows_x2,
    }
    out_path = os.path.join(args.out_dir, "s2_multi_x2_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    print("\n================ s2 multi summary ================")
    print(f"curves: {comp['n_curves']}")
    print(f"before: {b_before}  recall={b_before['ok'] / comp['n_curves']:.4f}")
    print(f"after : {b_after}  recall={b_after['ok'] / comp['n_curves']:.4f}")
    print(f"e1_2->ok: {comp['e1_2_to_ok']}  e1_2->e2_5: {comp['e1_2_to_e2_5']}  "
          f"ok->e1_2: {comp['ok_to_e1_2']}  e2_5->ok: {comp['e2_5_to_ok']}  "
          f"to_e5p: {comp['to_e5p']}  from_e5p: {comp['from_e5p']}")
    print(f"better={comp['better_any']} worse={comp['worse_any']}")
    print(f"wrote: {out_path}")
    return 0


def _prep_one(stem: str, out_dir: str) -> None:
    """Upsample one image + scale/copy its sidecars (mirrors prep script)."""
    import cv2

    p = stem + ".png"
    base = os.path.basename(stem)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    up = cv2.resize(img, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(os.path.join(out_dir, base + ".png"), up)
    for ext in (".csv", "_meta.json", "_curves.json"):
        src = stem + ext
        if os.path.exists(src):
            with open(src, "rb") as fi, open(os.path.join(out_dir, base + ext), "wb") as fo:
                fo.write(fi.read())
    for csv_path in glob.glob(stem + "_c*.csv"):
        cb = os.path.basename(csv_path)
        with open(csv_path, "rb") as fi, open(os.path.join(out_dir, cb), "wb") as fo:
            fo.write(fi.read())
    src_l = stem + "_labels.json"
    if os.path.exists(src_l):
        prep.scale_boxes(src_l, os.path.join(out_dir, base + "_labels.json"))
    src_m = stem + "_mcg.json"
    if os.path.exists(src_m):
        prep.scale_mcg(src_m, os.path.join(out_dir, base + "_mcg.json"))


if __name__ == "__main__":
    sys.exit(main())
