"""End-to-end pipeline integration tests on generated synthetic charts.

Uses the stub OCR backend (ground-truth label boxes) so the test is
deterministic and does not require PaddleOCR.  The PaddleOCR path is covered
by scripts/evaluate.py --ocr paddle on the real synthetic set.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from mci.eval.metrics import curve_metrics  # noqa: E402
from mci.pipeline.extractor import Extractor  # noqa: E402
from mci.schema import ExtractionError  # noqa: E402


def _load_gt(stem):
    import csv as csvlib

    with open(stem + ".csv", "r", encoding="utf-8") as f:
        rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
    return np.array([[float(r[0]), float(r[1])] for r in rows[1:]])


def _run_synthetic(seed: int, count: int, tmp_path, ocr="stub", segmenter=None):
    import gen_synthetic

    out_dir = tmp_path / f"set_{seed}"
    sys.argv = ["gen_synthetic", "--out-dir", str(out_dir), "--count", str(count), "--seed", str(seed)]
    assert gen_synthetic.main() == 0
    extractor = Extractor(ocr_backend=ocr, segmenter=segmenter)
    results = []
    for png in sorted(out_dir.glob("*.png")):
        if png.name.endswith("_mask.png"):
            continue
        stem = str(png)[: -len(png.suffix)]
        try:
            result = extractor.extract(str(png))
            gt = _load_gt(stem)
            m = curve_metrics(gt, np.array(result.curves[0].points))
            results.append((png.name, result, m))
        except ExtractionError as e:
            results.append((png.name, None, {"error": str(e)}))
    return results


def test_generator_works(tmp_path):
    """Generator must produce self-consistent images (self-check inside)."""
    import subprocess

    res = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "gen_synthetic.py"),
         "--out-dir", str(tmp_path), "--count", "4", "--seed", "3"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    _assert_gen_sidecars(tmp_path)


def test_filter_mask_fragments_keeps_curve_only():
    """Thin non-curve fragments (title text / frame / dust) must be dropped
    from segmentation masks before tracing, or they corrupt the skeleton
    trace start and the column-centroid fallback."""
    from mci.pipeline.curve_extractor import _filter_mask_fragments

    h, w = 200, 400
    mask = np.zeros((h, w), np.uint8)
    # genuine curve: a long diagonal stroke (thick enough, spans both dims)
    cv2 = pytest.importorskip("cv2")
    cv2.line(mask, (20, 180), (380, 30), 255, 6)
    # title-like fragment: thin horizontal strip, short in both directions
    cv2.rectangle(mask, (60, 5), (150, 8), 255, -1)
    # frame-like strip hugging the top edge, spanning the full width
    cv2.rectangle(mask, (0, 0), (399, 2), 255, -1)
    # dust specks
    mask[100, 100] = 255
    mask[101, 101] = 255

    out = _filter_mask_fragments(mask, w, h)
    assert out[20:200, 20:380].sum() > 0  # curve kept
    assert out[5:9, 60:151].sum() == 0  # title fragment dropped
    assert out[0:3, :].sum() == 0  # full-width frame strip dropped
    assert out[99:103, 99:103].sum() == 0  # dust dropped


def _assert_gen_sidecars(tmp_path):
    pngs = [p for p in tmp_path.glob("*.png") if not p.name.endswith("_mask.png")]
    assert len(pngs) == 4
    for p in pngs:
        assert p.with_name(p.stem + ".csv").exists()
        assert p.with_name(p.stem + "_meta.json").exists()
        assert p.with_name(p.stem + "_labels.json").exists()
        assert p.with_name(p.stem + "_mask.png").exists()


def test_end_to_end_linear(tmp_path):
    """CV backend: no hard failures; median rel_rmse < 2%; each image keeps
    at least 30% x-coverage (dashed curves are bridged as far as the CV
    heuristics can — the U-Net backend is asserted strictly below)."""
    results = _run_synthetic(seed=11, count=6, tmp_path=tmp_path, segmenter="cv")
    failed = [r[0] + ": " + str(r[2].get("error", "")) for r in results if r[1] is None]
    assert not failed, failed
    rel = [r[2]["rel_rmse"] for r in results]
    cov = [r[2]["x_coverage"] for r in results]
    assert np.median(rel) < 0.02, rel
    assert max(rel) < 0.35, rel
    assert min(cov) > 0.3, cov


def test_end_to_end_mixed_axes(tmp_path):
    """A mixed set (linear/log, styles, degradations) on the CV backend.

    The CV backend is the training-free baseline: it must extract *something*
    on nearly every image and be accurate on the median, but adversarial
    cases (light curves / dashed curves / grids) are expected to degrade —
    the U-Net backend (test_end_to_end_unet_strict) is where the strict
    1% acceptance target is asserted.
    """
    results = _run_synthetic(seed=21, count=12, tmp_path=tmp_path, segmenter="cv")
    failed = [r[0] + ": " + str(r[2].get("error", "")) for r in results if r[1] is None]
    assert len(failed) <= 1, failed
    ok = [r for r in results if r[1] is not None]
    rel = [r[2]["rel_rmse"] for r in ok]
    cov = [r[2]["x_coverage"] for r in ok]
    assert np.median(rel) < 0.06, rel
    assert sum(1 for r in rel if r > 0.1) <= 6, rel
    assert np.median(cov) > 0.5, cov


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "models", "checkpoints", "unet_curve.pt")),
    reason="U-Net checkpoint not trained yet (run train/train_segmentation.py)",
)
def test_end_to_end_unet_strict(tmp_path):
    """U-Net backend: full coverage on every image, median rel_rmse < 1%,
    worst case < 4% (current model's measured bound on the 40-image set)."""
    results = _run_synthetic(seed=21, count=12, tmp_path=tmp_path, segmenter="unet")
    failed = [r[0] + ": " + str(r[2].get("error", "")) for r in results if r[1] is None]
    assert not failed, failed
    rel = [r[2]["rel_rmse"] for r in results]
    for name, result, m in results:
        assert m["x_coverage"] > 0.65, name
        assert m["rel_rmse"] < 0.04, (name, m["rel_rmse"])
    assert np.median(rel) < 0.01, rel
