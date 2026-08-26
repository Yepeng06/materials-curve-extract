"""Validate gold annotations (WebPlotDigitizer exports) against the project
format required by the evaluation scripts (real_papers README §3.3).

Expected layout under a gold directory (one subdir or flat per image stem):

    <stem>_curves.json    {"curves": [{"label": ..., "csv": "<stem>_c1.csv"}, ...]}
    <stem>_c1.csv         x,y point table (header row optional)
    <stem>_meta.json      {"x_kind", "y_kind", "x_range", "y_range", "n_curves"}

Usage:
  python scripts/validate_gold_csv.py --gold data/real_papers/gold
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys


def validate_stem(gold_dir: str, stem: str) -> list:
    issues: list = []
    curves_json = os.path.join(gold_dir, stem + "_curves.json")
    meta_json = os.path.join(gold_dir, stem + "_meta.json")
    if not os.path.exists(curves_json):
        issues.append(f"{stem}: missing {stem}_curves.json")
        return issues
    with open(curves_json, encoding="utf-8-sig") as f:
        data = json.load(f)
    curves = data.get("curves", [])
    if not curves:
        issues.append(f"{stem}: _curves.json has no curves")
    for c in curves:
        csv_name = c.get("csv")
        if not csv_name:
            issues.append(f"{stem}: curve entry without csv filename")
            continue
        csv_path = os.path.join(gold_dir, csv_name)
        if not os.path.exists(csv_path):
            issues.append(f"{stem}: missing {csv_name} (declared in _curves.json)")
            continue
        n_pts, bad = 0, 0
        with open(csv_path, encoding="utf-8-sig") as f2:
            for row in csv.reader(f2):
                if not row or row[0].startswith("#"):
                    continue
                if row[0].lower() in ("x", "t", "time"):
                    continue
                try:
                    float(row[0]); float(row[1]); n_pts += 1
                except (ValueError, IndexError):
                    bad += 1
        if n_pts < 3:
            issues.append(f"{stem}/{csv_name}: only {n_pts} numeric points (need >= 3)")
        if bad:
            issues.append(f"{stem}/{csv_name}: {bad} unparsable rows")
    if os.path.exists(meta_json):
        with open(meta_json, encoding="utf-8-sig") as f:
            meta = json.load(f)
        for k in ("x_kind", "y_kind", "x_range", "y_range"):
            if k not in meta:
                issues.append(f"{stem}: meta missing '{k}'")
        if meta.get("n_curves") is not None and meta["n_curves"] != len(curves):
            issues.append(f"{stem}: meta n_curves={meta['n_curves']} != curves {len(curves)}")
    else:
        issues.append(f"{stem}: missing {stem}_meta.json (recommended)")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=os.path.join("data", "real_papers", "gold"))
    args = ap.parse_args()

    stems = sorted({
        re.sub(r"_curves\.json$|_meta\.json$|_c\d+\.csv$", "", p)
        for p in os.listdir(args.gold)
        if re.search(r"_curves\.json$|_meta\.json$|_c\d+\.csv$", p)
    })
    if not stems:
        print(f"no annotations found in {args.gold}")
        return 1

    total_issues = 0
    for stem in stems:
        issues = validate_stem(args.gold, stem)
        if issues:
            total_issues += len(issues)
            print(f"[FAIL] {stem}")
            for i in issues:
                print(f"    - {i}")
        else:
            print(f"[ OK ] {stem}")
    print(f"\nstems: {len(stems)}  issues: {total_issues}")
    return 1 if total_issues else 0


if __name__ == "__main__":
    sys.exit(main())
