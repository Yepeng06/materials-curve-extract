"""Phase B-1 axis metrics: tick-text recognition rate + axis-type accuracy.

Reads an evaluate.py report.csv (which now carries per-axis GT/read tick
counts and GT/pred axis kinds) and prints aggregate statistics:
  * tick text recognition rate per axis (read/GT, capped at 1 per image),
  * coordinate type accuracy per axis (linear/log judgement),
  * pass rate and RMSE headline numbers from summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True,
                    help="path to evaluate.py report.csv")
    args = ap.parse_args()

    rows = []
    with open(args.report, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    n_img = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    n_ok = len(ok)

    def rate(key_gt, key_pred):
        tot = n_miss = 0
        for r in rows:
            g = r.get(key_gt)
            p = r.get(key_pred)
            if g is None or p is None or not g or not p:
                continue
            tot += 1
            if g != p:
                n_miss += 1
        return (tot - n_miss) / tot if tot else float("nan")

    def tick_rate(key_gt, key_read):
        total_gt = total_read = 0
        for r in rows:
            g = int(float(r[key_gt])) if r.get(key_gt) else 0
            rd = int(float(r[key_read])) if r.get(key_read) else 0
            total_gt += g
            total_read += min(rd, g)
        return total_read / total_gt if total_gt else float("nan")

    print(f"n_images: {n_img}  n_ok: {n_ok}  n_failed: {n_img - n_ok}")
    print(f"axis-type accuracy: x={rate('x_kind_gt', 'x_kind_pred'):.4f}  "
          f"y={rate('y_kind_gt', 'y_kind_pred'):.4f}")
    print(f"tick recognition rate: x={tick_rate('n_ticks_x_gt', 'n_ticks_x_read'):.4f}  "
          f"y={tick_rate('n_ticks_y_gt', 'n_ticks_y_read'):.4f}")

    # per-image read fraction (median) for the summary
    fracs = []
    for r in rows:
        gx = int(float(r["n_ticks_x_gt"])) if r.get("n_ticks_x_gt") else 0
        gy = int(float(r["n_ticks_y_gt"])) if r.get("n_ticks_y_gt") else 0
        rx = int(float(r["n_ticks_x_read"])) if r.get("n_ticks_x_read") else 0
        ry = int(float(r["n_ticks_y_read"])) if r.get("n_ticks_y_read") else 0
        if gx + gy > 0:
            fracs.append((rx + ry) / (gx + gy))
    if fracs:
        fracs.sort()
        med = fracs[len(fracs) // 2]
        print(f"per-image tick read fraction: median={med:.4f}  "
              f"min={fracs[0]:.4f}  n_full_read={sum(1 for f in fracs if f >= 1.0)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
