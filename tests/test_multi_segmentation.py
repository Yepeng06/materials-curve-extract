"""Phase C: multi-curve instance segmentation tests."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


# ---------------------------------------------------------------------------
# instance masks rebuilt from GT CSVs through the label mapping
# ---------------------------------------------------------------------------
def _mk_fixture(tmp_path, n_curves):
    import csv as _csv
    labels = [
        {"box": [[80, 542], [120, 542], [120, 562], [80, 562]], "text": "1", "score": 1.0},
        {"box": [[350, 542], [390, 542], [390, 562], [350, 562]], "text": "10", "score": 1.0},
        {"box": [[60, 500], [100, 500], [100, 520], [60, 520]], "text": "0", "score": 1.0},
        {"box": [[60, 300], [100, 300], [100, 320], [60, 320]], "text": "0.5", "score": 1.0},
    ]
    (tmp_path / "img_labels.json").write_text(json.dumps(labels), encoding="utf-8")
    curves = {"num_curves": n_curves, "curves": []}
    for i in range(n_curves):
        fn = f"img_c{i}.csv"
        (tmp_path / fn).write_text(
            "x,y\n" + "\n".join(f"{x},{0.1 * (i + 1) + 0.001 * x}" for x in range(1, 101)),
            encoding="utf-8",
        )
        curves["curves"].append({"curve_id": f"curve_{i}", "label": f"Curve {i}", "csv": fn})
    (tmp_path / "img_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    return curves


def test_meta_instance_masks_separates_curves(tmp_path):
    from train.train_segmentation_multi import _meta_instance_masks

    curves_json = _mk_fixture(tmp_path, 2)
    labels = json.loads((tmp_path / "img_labels.json").read_text(encoding="utf-8"))
    m = _meta_instance_masks({}, labels, curves_json, str(tmp_path), (400, 600), k=6)
    assert m.shape[0] == 6
    c0 = m[0] > 0
    c1 = m[1] > 0
    assert c0.sum() > 0 and c1.sum() > 0
    assert not (c0 & c1).any()  # separated instances
    assert (m[2] > 0).sum() == 0  # extra channels empty


def test_meta_instance_masks_single_curve(tmp_path):
    from train.train_segmentation_multi import _meta_instance_masks

    curves_json = _mk_fixture(tmp_path, 1)
    labels = json.loads((tmp_path / "img_labels.json").read_text(encoding="utf-8"))
    m = _meta_instance_masks({}, labels, curves_json, str(tmp_path), (400, 600), k=6)
    assert (m[0] > 0).sum() >= 50
    assert (m[1] > 0).sum() == 0


# ---------------------------------------------------------------------------
# UNet out_channels
# ---------------------------------------------------------------------------
def test_unet_out_channels():
    import torch

    from mci.models.segmentation.unet import UNet

    model = UNet(in_channels=1, base=16, out_channels=6)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 6, 64, 64)
    # backward compatible: default stays single-channel
    m1 = UNet(in_channels=1, base=16)
    assert m1(x).shape == (2, 1, 64, 64)


# ---------------------------------------------------------------------------
# multi-curve extraction
# ---------------------------------------------------------------------------
def _fake_multi_segmenter(mask01):
    class S:
        def prob_full(self, image_bgr):
            h, w = image_bgr.shape[:2]
            out = np.zeros((6, h, w), np.float32)
            # two curve instances at different rows
            out[0, 40:60, 20:180] = 0.9
            out[1, 140:160, 20:180] = 0.9
            return out
    return S()


def test_extract_curves_multi_separates_instances():
    import cv2

    from mci.pipeline.curve_extractor import extract_curves_multi
    from mci.schema import AxisKind, AxisRole, AxisSpec, ChartStructure

    img = np.full((300, 200, 3), 255, np.uint8)
    structure = ChartStructure(plot_bbox=(10, 10, 190, 290),
                               x_axis_pixel=290, y_axis_pixel=10,
                               x_ticks_px=[], y_ticks_px=[])
    x_axis = AxisSpec(role=AxisRole.X, kind=AxisKind.LINEAR, slope=1.0,
                      intercept=0.0, vmin=0, vmax=100, pmin=10, pmax=190,
                      sign=1, ticks=[], quality=1.0)
    y_axis = AxisSpec(role=AxisRole.Y, kind=AxisKind.LINEAR, slope=-1.0,
                      intercept=300.0, vmin=0, vmax=100, pmin=10, pmax=290,
                      sign=-1, ticks=[], quality=1.0)
    curves = extract_curves_multi(img, structure, x_axis, y_axis, {},
                                  _fake_multi_segmenter(None))
    assert len(curves) == 2
    y_means = [float(np.mean([p[1] for p in c.points])) for c in curves]
    assert max(y_means) - min(y_means) > 10  # two well-separated curves

# ---------------------------------------------------------------------------
# jump truncation (regression: off-by-one on steep-tail chains, 2026-08-20)
# ---------------------------------------------------------------------------
def _chain(n_flat, n_steep=2, jump=60.0):
    """n_flat gentle points then a steep sustained tail (the crash case)."""
    pts = [(float(i), float(50 + 0.2 * i)) for i in range(n_flat)]
    y = pts[-1][1]
    for k in range(n_steep):
        y += jump
        pts.append((float(n_flat + k), y))
    return pts


def test_truncate_jumps_steep_tail_no_crash():
    from mci.pipeline.curve_extractor import _truncate_jumps

    # two consecutive 60px tail jumps: the run is cut (no IndexError)
    ch = _chain(120, n_steep=2)
    out = _truncate_jumps(ch)
    assert len(out) == len(ch) - 2     # jumped tail points removed
    dy = np.abs(np.diff([p[1] for p in out]))
    assert float(dy[-1]) <= 30.0       # no jump left at the end
    # long sustained tail: last two points removed, no crash
    ch6 = _chain(120, n_steep=6)
    out6 = _truncate_jumps(ch6)
    assert len(out6) == len(ch6) - 2
    # single isolated jump in the middle (no consecutive pair): unchanged
    mid = [(float(i), float(50 + 0.2 * i)) for i in range(200)]
    mid[100] = (100.0, 50.0 + 60.0)          # one jump, then flat from new level
    for i in range(101, 200):
        mid[i] = (float(i), 110.0 + 0.2 * (i - 100))
    assert _truncate_jumps(mid) == mid


def test_truncate_jumps_gentle_chain_unchanged():
    from mci.pipeline.curve_extractor import _truncate_jumps

    ch = [(float(i), float(50 + 0.2 * i)) for i in range(200)]
    assert _truncate_jumps(ch) == ch


def test_truncate_jumps_short_chain_unchanged():
    from mci.pipeline.curve_extractor import _truncate_jumps

    ch = [(float(i), float(i)) for i in range(30)]  # < min_len + 2
    assert _truncate_jumps(ch) == ch

