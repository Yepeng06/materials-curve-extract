"""CLI: batch evaluation of the baseline on a synthetic test set.

Usage:
  python scripts/evaluate.py --data-dir data/synthetic [--out-dir data/eval]
                             [--config configs/baseline.yaml] [--ocr paddle|stub|auto]
                             [--limit N] [--save-debug]

Requires the generator's ground truth files next to each image
(<stem>.csv, <stem>_meta.json).  Reports per-image metrics (RMSE / relative
RMSE / coverage) and an aggregate summary; the acceptance target is
rel_rmse <= 1% of the y-axis full scale.
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np  # noqa: E402

from mci.eval.metrics import curve_metrics, summarize  # noqa: E402
from mci.pipeline.extractor import Extractor  # noqa: E402
from mci.schema import ExtractionError  # noqa: E402


def load_gt(stem: str) -> np.ndarray:
    with open(stem + ".csv", "r", encoding="utf-8") as f:
        rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
    return np.array([[float(r[0]), float(r[1])] for r in rows[1:]])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/synthetic")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ocr", default="auto", choices=["auto", "paddle", "stub"])
    ap.add_argument("--segmenter", default=None, choices=[None, "cv", "unet"],
                    help="curve segmentation backend (default: config value)")
    ap.add_argument("--unet-size", type=int, default=None,
                    help="U-Net inference resolution (must match training; "
                         "default: config unet_size=256)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-debug", action="store_true")
    args = ap.parse_args()

    images = sorted(set(
        glob.glob(os.path.join(args.data_dir, "*.png"))
        + glob.glob(os.path.join(args.data_dir, "*.jpg"))
    ))
    images = [p for p in images if not os.path.basename(p).endswith("_mask.png")]
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"no images in {args.data_dir}")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    debug_dir = os.path.join(args.out_dir, "debug") if args.save_debug else None
    cfg_override = None
    if args.unet_size:
        cfg_override = {"unet_size": args.unet_size}
    extractor = Extractor(config_path=args.config, ocr_backend=args.ocr,
                          segmenter=args.segmenter, debug_dir=debug_dir,
                          config_override=cfg_override)

    rows = []
    for img_path in images:
        stem = os.path.splitext(img_path)[0]
        row = {"image": os.path.basename(img_path)}
        try:
            with open(stem + "_meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            row["x_kind_gt"] = meta["x_kind"]
            row["y_kind_gt"] = meta["y_kind"]
            gt = load_gt(stem)
            result = extractor.extract(img_path)
            pred = np.array(result.curves[0].points)
            m = curve_metrics(gt, pred)
            row.update(m)
            row["x_kind_pred"] = result.x_axis.kind.value
            row["y_kind_pred"] = result.y_axis.kind.value
            row["status"] = "ok"
            row["timing_s"] = round(result.meta["timings"]["total"], 3)
            row["warnings"] = "; ".join(result.warnings)
            print(f"[ OK ] {row['image']}: rel_rmse={m['rel_rmse']:.5f} "
                  f"cov={m['x_coverage']:.3f} axes=({row['x_kind_pred']},{row['y_kind_pred']})")
        except ExtractionError as e:
            row["status"] = "failed"
            row["error"] = str(e)[:200]
            print(f"[FAIL] {row['image']}: {e}")
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[ERR ] {row['image']}: {e}\n{traceback.format_exc()}")
        rows.append(row)

    summary = summarize(rows)
    rep_path = os.path.join(args.out_dir, "report.csv")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        writer = csvlib.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ summary ================")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"report: {rep_path}")
    return 0 if summary["n_failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
