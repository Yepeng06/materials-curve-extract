"""Trainable curve segmenter (U-Net) and CV fallback — same output contract.

``segment_plot_region(image_bgr, structure, ...) -> np.ndarray`` returns a
0/1 mask of the plot region whose foreground is the curve.  Two backends:

* ``UNetSegmenter`` — learned (train/train_segmentation.py); robust to light
  curves, dashed curves, grids, JPEG noise.
* classical CV (binarization + grid removal + component selection) — inside
  ``curve_extractor``; kept as the training-free fallback.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import torch

from ..schema import ChartStructure, CurveExtractionError
from ..utils import ink_mask


class UNetSegmenter:
    """U-Net curve segmentation with full-image inference."""

    def __init__(self, checkpoint: str, device: str = "auto", size: int = 256):
        self.size = size
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if not os.path.exists(checkpoint):
            raise CurveExtractionError(
                f"unet checkpoint not found: {checkpoint} (run train/train_segmentation.py)"
            )
        from ..models.segmentation.unet import UNet

        self.model = UNet(in_channels=1, base=64).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def segment_full(self, image_bgr: np.ndarray) -> np.ndarray:
        """0/1 mask at full image size (1 = curve)."""
        return (self.prob_full(image_bgr) > 0.5).astype(np.uint8)

    def prob_full(self, image_bgr: np.ndarray) -> np.ndarray:
        """Float curve-probability map at full image size (sub-pixel usable)."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            logit = self.model(x.to(self.device))
        prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class MultiUNetSegmenter:
    """K-channel instance U-Net (Phase C): one probability map per curve.

    ``prob_full`` returns a (K, H, W) float tensor; channel c = probability
    that pixel belongs to curve instance c (empty channels for charts with
    fewer than K curves).
    """

    def __init__(self, checkpoint: str, device: str = "auto", size: int = 256,
                 out_channels: int = 6):
        self.size = size
        self.out_channels = out_channels
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if not os.path.exists(checkpoint):
            raise CurveExtractionError(
                f"multi-unet checkpoint not found: {checkpoint} (run "
                f"train/train_segmentation_multi.py)"
            )
        from ..models.segmentation.unet import UNet

        self.model = UNet(in_channels=1, base=64,
                          out_channels=out_channels).to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def prob_full(self, image_bgr: np.ndarray) -> np.ndarray:
        """(K, H, W) float instance-probability maps at full image size."""
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            logit = self.model(x.to(self.device))
        prob = torch.sigmoid(logit)[0].cpu().numpy()  # (K, size, size)
        out = np.stack([cv2.resize(prob[c], (w, h), interpolation=cv2.INTER_LINEAR)
                        for c in range(prob.shape[0])]).astype(np.float32)
        return out


def cv_plot_mask(image_bgr: np.ndarray, structure: ChartStructure,
                 cfg: Optional[dict] = None) -> np.ndarray:
    """Classical-CV mask for the plot region (grid removal included)."""
    from .curve_extractor import _bridge_dash_gaps, _remove_gridlines

    x0, y0, x1, y1 = structure.plot_bbox
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = ink_mask(gray)
    bw = 3
    ink[:bw, :] = 0
    ink[-bw:, :] = 0
    ink[:, :bw] = 0
    ink[:, -bw:] = 0
    ink = _remove_gridlines(ink, image_bgr, structure, cfg or {})
    ink = _bridge_dash_gaps(ink)
    return ink
