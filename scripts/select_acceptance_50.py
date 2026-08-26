"""S4: select a ~50-image acceptance subset from the 326-image creep library.

Stratified by difficulty (easy/medium/hard) with diversity constraints
(material class, paper DOI, axis kind, curve count).  Outputs
``data/real_papers/manifest.csv`` — the manifest template for the
real-figure acceptance run (REAL_ROBUSTNESS_DESIGN.md S4 / real_papers README).

Usage:
  python scripts/select_acceptance_50.py [--n-easy 15] [--n-medium 20] [--n-hard 15]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

CREEP_DB = r"F:\CODE\New\dataset\creep_curves_dataset.json"
PANEL_CSV = r"F:\CODE\New\dataset\panel_estimate_v2.csv"
OUT = os.path.join("data", "real_papers", "manifest.csv")

# preferred plot types for the acceptance set (A family first)
_PLOT_PRIORITY = ["A", "A2", "A3", "D", "E", "B", "C", "F", "H"]


def load_db() -> list:
    with open(CREEP_DB, encoding="utf-8") as f:
        return json.load(f)


def load_panels() -> dict:
    out = {}
    if not os.path.exists(PANEL_CSV):
        return out
    with open(PANEL_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["image"]] = int(r["est_panels"])
    return out


def axis_kind_of(rec: dict) -> str:
    x = (rec.get("x_label") or "").lower()
    y = (rec.get("y_label") or "").lower()
    xk = "log" if "log(" in x or x.startswith("log") else "lin"
    yk = "log" if "log(" in y or y.startswith("log") else "lin"
    return f"{xk}/{yk}"


def difficulty_of(rec: dict, panels: int) -> str:
    n = int(rec.get("n_curves") or 1)
    ax = axis_kind_of(rec)
    if panels >= 3 or n >= 5 or ax == "log/log":
        return "hard"
    if panels == 2 or n >= 3 or ax in ("log/lin", "lin/log") or not rec.get("complete", True):
        return "medium"
    return "easy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-easy", type=int, default=15)
    ap.add_argument("--n-medium", type=int, default=20)
    ap.add_argument("--n-hard", type=int, default=15)
    ap.add_argument("--min-log", type=int, default=10,
                    help="minimum number of log-axis images (Norton D class "
                         "counts, and labels containing 'Log')")
    args = ap.parse_args()

    db = load_db()
    panels = load_panels()

    # only images that exist on disk
    base = r"F:\CODE\New\dataset\curves_final"
    recs = [r for r in db if os.path.exists(os.path.join(base, r["image"]))]
    print(f"library records on disk: {len(recs)}")

    for r in recs:
        p = panels.get(r["image"], 0)
        r["_panels"] = p
        r["_axis"] = axis_kind_of(r)
        r["_difficulty"] = difficulty_of(r, p)
        # log-axis signal: Norton double-log (D) or explicit Log labels
        r["_log"] = r["_axis"].startswith("log") or r["_axis"].endswith("/log") \
            or r.get("plot_type") == "D"

    picked: list = []

    def _tier_count(tier: str) -> int:
        return {"easy": args.n_easy, "medium": args.n_medium,
                "hard": args.n_hard}[tier]

    def _pick(tier: str, target: int, log_only: bool = False) -> None:
        cand = [r for r in recs
                if r["_difficulty"] == tier and r not in picked
                and (not log_only or r["_log"])]
        cand.sort(key=lambda r: (
            _PLOT_PRIORITY.index(r.get("plot_type")) if r.get("plot_type") in _PLOT_PRIORITY else 99,
            r.get("paper"),
            -int(r.get("n_curves") or 1),
        ))
        for r in cand:
            if len([x for x in picked if x["_difficulty"] == tier]) >= target:
                break
            if len([x for x in picked if x.get("paper") == r.get("paper")]) >= 2:
                continue
            picked.append(r)

    # pass 1: log-axis quota first, spread across tiers (easy 2 / medium 4 /
    # hard 4 when min-log=10); pass 2 tops each tier up to its full target.
    log_quota = {
        "easy": max(2, args.min_log // 5),
        "medium": max(4, args.min_log * 2 // 5),
        "hard": max(4, args.min_log - args.min_log // 5 - args.min_log * 2 // 5),
    }
    for tier in ("easy", "medium", "hard"):
        _pick(tier, log_quota[tier], log_only=True)
    for tier in ("easy", "medium", "hard"):
        _pick(tier, _tier_count(tier), log_only=False)

    picked.sort(key=lambda r: (r["_difficulty"], r["image"]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "difficulty", "plot_type", "axis", "n_curves",
                    "material", "paper", "est_panels"])
        for r in picked:
            w.writerow([r["image"], r["_difficulty"], r.get("plot_type"),
                        r["_axis"], r.get("n_curves"), r.get("material"),
                        r.get("paper"), r["_panels"]])

    from collections import Counter
    print(f"picked: {len(picked)}")
    print("difficulty:", dict(Counter(r["_difficulty"] for r in picked)))
    print("axis:", dict(Counter(r["_axis"] for r in picked)))
    print(f"log-related: {sum(1 for r in picked if r['_log'])}")
    print("plot_type:", dict(Counter(r.get("plot_type") for r in picked)))
    print("materials:", len({r.get("material") for r in picked}),
          "papers:", len({r.get("paper") for r in picked}))
    print("saved:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
