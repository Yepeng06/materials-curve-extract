"""Phase C: multi-curve instance segmentation tests."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


# ---------------------------------------------------------------------------
# instance masks from meta.curves_px polylines
# ---------------------------------------------------------------------------
def test_meta_instance_masks_separates_curves():
    from train.train_segmentation_multi import _meta_instance_masks

    meta = {"curves_px": [
        [[20.0, 30.0], [40.0, 30.0], [60.0, 30.0], [80.0, 30.0]],
        [[20.0, 70.0], [40.0, 70.0], [60.0, 70.0], [80.0, 70.0]],
    ]}
    m = _meta_instance_masks(meta, (100, 100), k=6)
    assert m.shape[0] == 6
    c0 = m[0] > 0
    c1 = m[1] > 0
    assert c0.sum() > 0 and c1.sum() > 0
    assert not (c0 & c1).any()  # separated instances
    assert (c0[30].sum() > 0) and (c1[70].sum() > 0)
    assert (m[2] > 0).sum() == 0  # extra channels empty


def test_meta_instance_masks_single_curve():
    from train.train_segmentation_multi import _meta_instance_masks

    meta = {"curves_px": [[[10.0, 50.0], [50.0, 50.0], [90.0, 50.0]]]}
    m = _meta_instance_masks(meta, (100, 100), k=6)
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
