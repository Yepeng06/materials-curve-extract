"""Curve segmentation U-Net (PyTorch).

Trains on synthetic chart images to segment the *curve* channel only —
grid lines, axes, text and legend boxes are all background.  This is the
trainable core of the baseline (per the project's technical route: U-Net +
post-processing) and replaces the classical ink-binarization + grid removal
steps in ``curve_extractor`` while the skeletonize / trace / coordinate
mapping stages stay shared.

Design notes (following APEX-Net / ChartOCR style chart-extraction work):
  * small U-Net (~2.1M params), 256x256 input, 1-channel grayscale;
  * BatchNorm + ReLU double convs, 3 down/up levels, skip connections;
  * trained with BCE + soft-Dice on full chart images, so the model learns
    to ignore axes/grid/text by construction (GT masks contain only the
    curve), which makes the whole extraction robust to light curves, dashed
    curves, grids and JPEG noise without any hand-tuned thresholds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """2D U-Net.  Output channels: 1 = single curve probability (baseline),
    K > 1 = per-instance curve channels (Phase C multi-curve)."""

    def __init__(self, in_channels: int = 1, base: int = 64,
                 out_channels: int = 1, embed_dim: int = 0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)
        # GOI embedding head (方案 E): parallel per-pixel embedding for
        # instance separation; forward() is unchanged for backward
        # compatibility, forward_embed() returns (logit, emb).
        self.embed_head = None
        if embed_dim > 0:
            self.embed_head = nn.Conv2d(base, embed_dim, 1)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1

    def forward(self, x):
        return self.out(self._encode(x))

    def forward_embed(self, x):
        """(logit, embedding) — embedding is None when embed_dim == 0."""
        d1 = self._encode(x)
        logit = self.out(d1)
        emb = self.embed_head(d1) if self.embed_head is not None else None
        return logit, emb


def bce_dice_loss(logit: torch.Tensor, target: torch.Tensor,
                  weight: torch.Tensor = None) -> torch.Tensor:
    """BCE + soft-Dice (target is 0/1 float tensor).

    ``weight``: optional per-pixel weight (B, 1, H, W) broadcastable to
    the logit shape; applied to the BCE term only (Dice stays uniform so
    the weighted region does not dominate the union statistics).
    """
    bce = F.binary_cross_entropy_with_logits(logit, target)
    if weight is not None:
        # per-pixel weighted BCE (reweighted mean, keeps the scale sane)
        bce_px = F.binary_cross_entropy_with_logits(
            logit, target, reduction="none")
        # weight is (1, 1, 1, W) broadcast over (B, K, H, W): the total
        # weight mass is weight.sum() * B * K * H
        wsum = weight.sum() * logit.shape[0] * logit.shape[1] * logit.shape[2]
        bce = (bce_px * weight).sum() / wsum.clamp(min=1e-6)
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    dice = 1.0 - (2.0 * inter + 1e-6) / union
    return bce + dice.mean()


def dice_iou(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
    """(dice, iou) per sample, thresholded."""
    p = (prob > thr).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = (p + target).clamp(max=1).sum(dim=(1, 2, 3)) + 1e-6
    dice = (2 * inter + 1e-6) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6)
    iou = inter / union
    return dice.mean().item(), iou.mean().item()
