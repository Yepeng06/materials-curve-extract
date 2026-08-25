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

# ---------------------------------------------------------------------------
# augmentation flip regression (2026-08-20: cv2.flip on (K,H,W) was a vertical
# mirror at size 512 and crashed at 768; the fix mirrors width + reverses
# channel order to stay consistent with the horizontally-mirrored image)
# ---------------------------------------------------------------------------
def test_augment_flip_mirrors_masks_with_image():
    import random
    from train.train_segmentation_multi import _augment

    rng_vals = iter([0.49, 0.9, 0.9, 0.9, 0.9, 0.9])  # flip ON, others OFF
    random.random = lambda: next(rng_vals)
    img = np.full((64, 64), 200, np.uint8)
    img[10, 10] = 0  # distinctive mark
    inst = np.zeros((4, 64, 64), np.uint8)
    inst[0, 10, 10] = 255  # channel 0 holds the mark
    skel = inst.copy()
    oi, om, os_, oc, ocx = _augment(img, inst, skel)
    assert oi[10, 63 - 10] == 0          # image horizontally mirrored
    assert om[3, 10, 63 - 10] == 255     # mask mirrored + channel reversed (0 -> 3)
    assert os_[3, 10, 63 - 10] == 255    # skeleton follows the mask
    assert oc is None                    # chain target None stays None


def test_augment_no_flip_keeps_inputs():
    import random
    from train.train_segmentation_multi import _augment

    rng_vals = iter([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])  # all augs OFF
    random.random = lambda: next(rng_vals)
    img = np.full((64, 64), 200, np.uint8)
    img[10, 10] = 0
    inst = np.zeros((4, 64, 64), np.uint8)
    inst[0, 10, 10] = 255
    oi, om, os_, oc, ocx = _augment(img, inst, inst.copy())
    assert oi[10, 10] == 0 and om[0, 10, 10] == 255 and os_[0, 10, 10] == 255
    assert oc is None

# ---------------------------------------------------------------------------
# 方案 E: GOI embedding head + loss + small-cluster merge (2026-08-20)
# ---------------------------------------------------------------------------
def test_unet_embed_head_forward_compat():
    import torch
    from mci.models.segmentation.unet import UNet

    m = UNet(in_channels=1, base=16, out_channels=6, embed_dim=8)
    x = torch.randn(2, 1, 32, 32)
    assert m(x).shape == (2, 6, 32, 32)          # forward unchanged
    logit, emb = m.forward_embed(x)
    assert logit.shape == (2, 6, 32, 32) and emb.shape == (2, 8, 32, 32)
    m0 = UNet(in_channels=1, base=16, out_channels=6)  # embed_dim=0
    logit0, emb0 = m0.forward_embed(x)
    assert emb0 is None


def test_goi_loss_pull_ortho():
    import torch
    from train.train_segmentation_multi import goi_loss

    B, E, H, W, K = 1, 8, 16, 16, 3
    y = torch.zeros(B, K, H, W)
    y[0, 0, 2:8, 2:8] = 1
    y[0, 1, 9:15, 9:15] = 1
    emb = torch.zeros(B, E, H, W)
    emb[0, 0, 2:8, 2:8] = 1.0    # instance 0 embedding = e_0
    emb[0, 1, 9:15, 9:15] = 1.0  # instance 1 embedding = e_1 (orthogonal)
    emb[0, 2, :, :] = 0.0        # empty channel
    emb = emb + torch.randn_like(emb) * 0.001
    pull, ortho = goi_loss(emb, y)
    assert pull.item() < 0.05      # intra-class pull ~0
    assert ortho.item() < 0.05     # orthogonal centroids ~0
    emb2 = emb.clone()
    emb2[0, 1, 9:15, 9:15] = 0.0   # strip e_1 from instance 1 pixels
    emb2[0, 0, 9:15, 9:15] = 1.0   # both instances now embed to e_0 -> collinear
    _, ortho2 = goi_loss(emb2, y)
    assert ortho2.item() > 0.5


def test_embed_merge_reassigns_small_component():
    from mci.pipeline.curve_extractor import _embed_merge_masks

    K, H, W, E = 2, 40, 40, 4
    masks = [np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)]
    masks[0][5:15, 5:15] = 1     # instance 0 block
    masks[1][25:35, 25:35] = 1   # instance 1 block
    reg = np.stack([m.astype(np.float32) * 0.9 for m in masks])
    emb = np.zeros((E, H, W), np.float32)
    emb[0, 5:15, 5:15] = 1.0      # instance 0 centroid = e_0
    emb[1, 25:35, 25:35] = 1.0    # instance 1 centroid = e_1
    emb[0] += 0.001; emb[1] += 0.001
    # small component (9px) inside instance 1's mask, embedding = e_1
    masks[1][30:33, 3:6] = 1
    emb[1, 30:33, 3:6] = 1.0
    out = _embed_merge_masks(masks, reg, emb, min_area=450, plot_w=W, plot_h=H)
    assert out[1][30:33, 3:6].sum() == 9   # stays in instance 1
    # same small comp but embedding = e_0 -> moves to instance 0
    masks[1][2:5, 2:5] = 1
    emb[0, 2:5, 2:5] = 1.0
    out2 = _embed_merge_masks(masks, reg, emb, min_area=450, plot_w=W, plot_h=H)
    assert out2[1][2:5, 2:5].sum() == 0     # moved away from instance 1
    assert out2[0][2:5, 2:5].sum() == 9     # into instance 0


