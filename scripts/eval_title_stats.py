"""Phase B-2 verification: title / axis label / unit parse rates vs GT.

Reads extractor outputs (meta.titles in result JSON) — run via a batch
script that saves per-image results, then compare with _meta.json GT.
"""
import argparse
import csv
import json
import os


def norm(s):
    return "".join(str(s).lower().split()).replace("-", "").replace(".", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.report, encoding="utf-8")))
    n = len(rows)
    stats = {r: [0, 0] for r in ("title", "x_label", "y_label")}
    var_ok = unit_ok = 0
    var_tot = unit_tot = 0

    def rate(key):
        hit = sum(1 for r in rows if r.get(key) == "1")
        return hit / n if n else 0.0

    for r in rows:
        for key in ("title", "x_label", "y_label"):
            stats[key][0] += 1 if r.get(key) == "1" else 0
        if r.get("x_var") == "1":
            var_ok += 1
        if r.get("y_var") == "1":
            var_ok += 1
        if r.get("x_unit") == "1":
            unit_ok += 1
        if r.get("y_unit") == "1":
            unit_ok += 1
        var_tot += 2
        unit_tot += 2

    print(f"n_images: {n}")
    for key, (hit, _) in stats.items():
        print(f"  {key} detection rate: {hit / n:.4f}")
    print(f"  variable parse rate (x+y): {var_ok / var_tot:.4f}")
    print(f"  unit parse rate (x+y): {unit_ok / unit_tot:.4f}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
