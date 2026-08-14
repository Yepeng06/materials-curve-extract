"""CLI: run the baseline extractor on one or more chart images.

Usage:
  python scripts/run_baseline.py --image path/to/chart.png [--out-dir data/outputs]
                                 [--config configs/baseline.yaml] [--ocr paddle|stub|auto]
                                 [--debug]

Outputs per image (in --out-dir):
  <stem>.csv      extracted curve data
  <stem>.json     full structured result
  <stem>_overlay.png  extraction visualized on the source image
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mci.export.csv_export import write_csv  # noqa: E402
from mci.export.json_export import write_json  # noqa: E402
from mci.export.visualizer import overlay_result  # noqa: E402
from mci.pipeline.extractor import Extractor  # noqa: E402
from mci.schema import ExtractionError  # noqa: E402
from mci.utils import read_image  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="chart image path (or --image-dir for batch)")
    ap.add_argument("--image-dir", help="directory of chart images (batch)")
    ap.add_argument("--pattern", default="*.png", help="glob pattern for --image-dir")
    ap.add_argument("--out-dir", default="data/outputs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ocr", default="auto", choices=["auto", "paddle", "stub"])
    ap.add_argument("--segmenter", default=None, choices=[None, "cv", "unet"],
                    help="curve segmentation backend (default: config value)")
    ap.add_argument("--debug", action="store_true", help="save per-step debug images")
    args = ap.parse_args()

    if args.image_dir:
        images = sorted(glob.glob(os.path.join(args.image_dir, args.pattern)))
        images += sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
        images += sorted(glob.glob(os.path.join(args.image_dir, "*.jpeg")))
        images = sorted(set(p for p in images if not os.path.basename(p).endswith("_mask.png")))
        if not images:
            print(f"no images found in {args.image_dir}")
            return 1
    elif args.image:
        images = [args.image]
    else:
        ap.error("provide --image or --image-dir")

    debug_dir = os.path.join(args.out_dir, "debug") if args.debug else None
    extractor = Extractor(config_path=args.config, ocr_backend=args.ocr,
                          segmenter=args.segmenter, debug_dir=debug_dir)

    for img_path in images:
        t0 = time.time()
        try:
            result = extractor.extract(img_path)
        except ExtractionError as e:
            print(f"[FAIL] {img_path}: {e}")
            continue
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_csv = write_csv(result, os.path.join(args.out_dir, f"{stem}.csv"))
        out_json = write_json(result, os.path.join(args.out_dir, f"{stem}.json"))
        overlay_result(result, read_image(img_path),
                       os.path.join(args.out_dir, f"{stem}_overlay.png"))
        n_pts = len(result.curves[0].points)
        print(f"[ OK ] {img_path}: {n_pts} pts, "
              f"x={result.x_axis.kind.value} y={result.y_axis.kind.value}, "
              f"{time.time() - t0:.2f}s -> {out_csv}")
        for wmsg in result.warnings:
            print(f"       warning: {wmsg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
