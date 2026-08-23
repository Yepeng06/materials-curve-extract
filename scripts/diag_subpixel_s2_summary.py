"""S2: aggregate the S1 (single decode compare) + S2 (2x upsampling) results
into data/experiments_s1s2/s2_upsample_summary.json.

Inputs (all produced by diag_subpixel_s1.py / diag_subpixel_s2_multi.py):
  data/experiments_s1s2/s1_decode_compare.json        (single, 1x)
  data/experiments_s1s2/s2_single_x2_decode_compare.json (single, 2x)
  data/experiments_s1s2/s2_multi_x2_result.json       (multi, 2x, selected imgs)
  data/eval_multi_diag_goi/diag.json                  (multi, 1x, full set)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

EXPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "experiments_s1s2")


def _load(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expdir", default=EXPDIR)
    args = ap.parse_args()
    d = args.expdir

    s1 = _load(os.path.join(d, "s1_decode_compare.json"))
    s2 = _load(os.path.join(d, "s2_single_x2_decode_compare.json"))
    multi_x2 = _load(os.path.join(d, "s2_multi_x2_result.json"))
    multi_base = _load(os.path.join(os.path.dirname(d), "eval_multi_diag_goi", "diag.json"))

    out = {
        "single_1x": s1["summary"],
        "single_2x": s2["summary"],
        "single_delta": {},
        "multi_full_baseline": multi_base["summary"],
        "multi_selected_1x": {
            "n_images": multi_x2["n_images_selected"],
            "n_curves": multi_x2["n_curves"],
            "buckets": multi_x2["buckets_before"],
            "recall": multi_x2["recall_before"],
        },
        "multi_selected_2x": {
            "n_images": multi_x2["n_images_selected"],
            "n_curves": multi_x2["n_curves"],
            "buckets": multi_x2["buckets_after"],
            "recall": multi_x2["recall_after"],
        },
        "multi_transitions": multi_x2["transitions"],
    }

    # per-decoder single-set delta 1x -> 2x
    for dn, s in s2["summary"].items():
        b = s1["summary"].get(dn)
        if not b:
            continue
        out["single_delta"][dn] = {
            "pass_rate_1pct_1x": b["pass_rate_1pct"],
            "pass_rate_1pct_2x": s["pass_rate_1pct"],
            "pass_delta_pp": (s["pass_rate_1pct"] - b["pass_rate_1pct"]) * 100.0,
            "median_rel_1x": b["median_rel"],
            "median_rel_2x": s["median_rel"],
            "buckets_1x": b["buckets"],
            "buckets_2x": s["buckets"],
            "bias_y_image_mean_1x": b["px_bias"]["image_mean_mean"],
            "bias_y_image_mean_2x": s["px_bias"]["image_mean_mean"],
            "bias_y_image_rmse_1x": b["px_bias"]["image_rmse_mean"],
            "bias_y_image_rmse_2x": s["px_bias"]["image_rmse_mean"],
        }

    out_path = os.path.join(d, "s2_upsample_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    print("================ single-set: pass rate / median rel (1x -> 2x) ================")
    for dn in ["centroid", "parabolic", "softargmax_T0.5", "softargmax_T0.25"]:
        delta = out["single_delta"].get(dn)
        if not delta:
            continue
        print(f"  [{dn}] pass {delta['pass_rate_1pct_1x']:.4f} -> {delta['pass_rate_1pct_2x']:.4f} "
              f"({delta['pass_delta_pp']:+.2f}pp)  median {delta['median_rel_1x']:.5f} -> "
              f"{delta['median_rel_2x']:.5f}  e1_2 {delta['buckets_1x']['e1_2']} -> "
              f"{delta['buckets_2x']['e1_2']}  e5p {delta['buckets_1x']['e5p']} -> {delta['buckets_2x']['e5p']}")
    print("\n================ multi selected set (1x -> 2x) ================")
    b1 = out["multi_selected_1x"]["buckets"]
    b2 = out["multi_selected_2x"]["buckets"]
    print(f"  curves={out['multi_selected_1x']['n_curves']}")
    print(f"  buckets 1x: ok={b1['ok']} e1_2={b1['e1_2']} e2_5={b1['e2_5']} e5p={b1['e5p']} "
          f"recall={out['multi_selected_1x']['recall']:.4f}")
    print(f"  buckets 2x: ok={b2['ok']} e1_2={b2['e1_2']} e2_5={b2['e2_5']} e5p={b2['e5p']} "
          f"recall={out['multi_selected_2x']['recall']:.4f}")
    t = out["multi_transitions"]
    print(f"  e1_2->ok={t['e1_2_to_ok']} e1_2->e2_5={t['e1_2_to_e2_5']} ok->e1_2={t['ok_to_e1_2']} "
          f"e2_5->ok={t['e2_5_to_ok']} to_e5p={t['to_e5p']} from_e5p={t['from_e5p']}")
    print(f"\nwrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
