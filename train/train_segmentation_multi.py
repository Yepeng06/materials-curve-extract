"""Train the multi-curve instance-segmentation U-Net (Phase C).

Usage (from the repo root, mci env):
  python train/train_segmentation_multi.py       --data-dir data/train_platform,data/train_platform_single       --val-dir data/val_multi,data/val_single       --epochs 60 --batch 8 --size 512       --init models/checkpoints/unet_curve.pt       --out models/checkpoints/unet_multi_curve.pt

Instance masks: K=6 channels (max curves).  Per-instance channels are
REBUILT from the GT CSVs through the tick-label mapping (labels.json),
which is the exact basis the evaluator compares against (validated 0.4px
mean agreement with the single-curve 512 model).  The network learns
instance separation; the channel order is the curves.json order per image
(LineFormer-style instance regression).
"""
from __future__ import annotations

import argparse
import csv as csvlib
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mci.models.segmentation.unet import UNet, bce_dice_loss, dice_iou

SIZE = 512
K = 6  # max curves per chart (train_platform has 2-5)


def goi_loss(emb: torch.Tensor, y: torch.Tensor) -> tuple:
    """GOI (Global Orthogonality Instance) loss — ChartZero-style (方案 E).

    emb: (B, E, H, W) raw per-pixel embeddings (L2-normalized here).
    y:   (B, K, H, W) 0/1 GT instance masks (overlapping crossing pixels
         excluded from the pull term — they are the ambiguous pixels the
         model must decide; the orthogonality term still separates the
         instances globally).

    Returns (pull, ortho) scalar losses, both empty-instance safe.
    """
    B, E, H, W = emb.shape
    Kc = y.shape[1]
    emb = torch.nn.functional.normalize(emb, dim=1)  # (B, E, H, W)
    ysum = y.sum(dim=1)                      # (B, H, W) instance count per pixel
    valid = (ysum > 0) & ~(ysum > 1)         # non-empty, non-overlap pixels
    labels = y.argmax(dim=1)                 # (B, H, W) instance label

    pull_num = torch.zeros((), device=emb.device)
    pull_den = torch.zeros((), device=emb.device)
    ortho_terms = []
    for b in range(B):
        cnts = []
        ctrs = []
        for k in range(Kc):
            m = y[b, k] > 0
            cnt = int(m.sum().item())
            cnts.append(cnt)
            if cnt > 0:
                c = emb[b, :, m].mean(dim=1)          # (E,)
                ctrs.append(torch.nn.functional.normalize(c, dim=0))
            else:
                ctrs.append(torch.zeros(E, device=emb.device))
        # pull over valid pixels of this sample
        m_valid = valid[b]
        if bool(m_valid.any()):
            lab = labels[b][m_valid]                   # (N,)
            px = emb[b, :, m_valid]                    # (E, N)
            cstack = torch.stack(ctrs, dim=1)          # (E, K)
            cos = (px * cstack[:, lab]).sum(dim=0)     # (N,)
            pull_num = pull_num + (1.0 - cos).sum()
            pull_den = pull_den + float(lab.numel())
        for i in range(Kc):
            for j in range(i + 1, Kc):
                if cnts[i] > 0 and cnts[j] > 0:
                    d = float((ctrs[i] * ctrs[j]).sum().item())
                    ortho_terms.append(d * d)
    pull = pull_num / pull_den.clamp(min=1e-6)
    ortho = (torch.tensor(ortho_terms, device=emb.device).mean()
             if ortho_terms else torch.zeros((), device=emb.device))
    return pull, ortho


def approach_zone_mask(y: torch.Tensor, dist: int = 4) -> torch.Tensor:
    """GT approach-zone pixels: pixels of instance i whose dilated mask
    touches instance j (curves closer than ``dist`` px).  (B, H, W) bool.

    P1a measured that 100% of the >5% failure error lies inside these
    zones and the embeddings/probabilities there are NOT discriminative
    (centroid cosine 0.9996) -- this loss gives the model an explicit
    training signal to separate instances there.
    """
    B, K, H, W = y.shape
    dil = torch.nn.functional.max_pool2d(
        y, kernel_size=2 * dist + 1, stride=1, padding=dist
    )  # (B, K, H, W) dilated instances
    overlap = (dil.sum(dim=1) > 1)  # pixels touched by >= 2 instances
    return overlap & (y.sum(dim=1) > 0)


def goi_loss_contrastive(emb: torch.Tensor, y: torch.Tensor,
                         near: torch.Tensor, margin: float = 0.2,
                         lam_pull: float = 1.0, lam_ortho: float = 1.0,
                         lam_contrast: float = 1.0) -> tuple:
    """GOI + approach-zone contrastive loss (杠杆 1).

    Standard GOI terms (pull, ortho) as in goi_loss, PLUS a contrastive
    term on approach-zone pixels: for every near pixel, push its embedding
    toward its OWN instance centroid and away from the WRONG instance
    centroid (hinge).  This directly targets the measured failure: the
    model currently cannot tell two curves apart where they touch.
    Returns (pull, ortho, contrast).
    """
    B, E, H, W = emb.shape
    Kc = y.shape[1]
    emb = torch.nn.functional.normalize(emb, dim=1)
    ysum = y.sum(dim=1)
    valid = (ysum > 0) & ~(ysum > 1)  # non-empty, non-overlap pixels
    labels = y.argmax(dim=1)

    # per-instance centroids from confident (non-near, non-overlap) pixels
    ctrs = torch.zeros(B, Kc, E, device=emb.device)
    ctr_n = torch.zeros(B, Kc, device=emb.device)
    for b in range(B):
        for k in range(Kc):
            m = (y[b, k] > 0) & valid[b] & ~near[b]
            if bool(m.any()):
                c = emb[b, :, m].mean(dim=1)
                ctrs[b, k] = torch.nn.functional.normalize(c, dim=0)
                ctr_n[b, k] = 1.0

    pull_num = torch.zeros((), device=emb.device)
    pull_den = torch.zeros((), device=emb.device)
    ortho_terms = []
    for b in range(B):
        m_valid = valid[b] & ~near[b]
        if bool(m_valid.any()):
            lab = labels[b][m_valid]
            px = emb[b, :, m_valid]                # (E, N)
            cos = (px.t() * ctrs[b][lab]).sum(dim=1)  # (N,)
            pull_num = pull_num + (1.0 - cos).sum()
            pull_den = pull_den + float(lab.numel())
        for i in range(Kc):
            for j in range(i + 1, Kc):
                if ctr_n[b, i] > 0 and ctr_n[b, j] > 0:
                    ortho_terms.append(float((ctrs[b, i] * ctrs[b, j]).sum().item()) ** 2)

    pull = pull_num / pull_den.clamp(min=1e-6)
    ortho = (torch.tensor(ortho_terms, device=emb.device).mean()
             if ortho_terms else torch.zeros((), device=emb.device))

    # ---- contrastive on approach-zone pixels ----
    contr_num = torch.zeros((), device=emb.device)
    contr_den = torch.zeros((), device=emb.device)
    if bool(near.any()):
        for b in range(B):
            m = near[b] & valid[b]
            if not bool(m.any()):
                continue
            lab = labels[b][m]                       # (N,)
            px = emb[b, :, m]                        # (E, N)
            cstack = ctrs[b]                         # (K, E)
            cos_all = cstack @ px                    # (K, N)
            cos_own = cos_all[lab, torch.arange(cos_all.shape[1], device=emb.device)]
            # wrong centroid = argmax over the OTHER instances
            cos_wrong, _ = cos_all.clone().scatter_(
                0, lab.unsqueeze(0), torch.full_like(cos_all, -2.0)
            ).max(dim=0)
            contr_num = contr_num + torch.clamp(cos_wrong - cos_own + margin, min=0.0).sum()
            contr_den = contr_den + float(lab.numel())
    contrast = contr_num / contr_den.clamp(min=1e-6)
    return pull, ortho, contrast


def _list_pairs(data_dir: str):
    pairs = []
    for d in data_dir.split(","):
        d = d.strip()
        if not d or not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.png"))):
            if p.endswith("_mask.png"):
                continue
            mask = p[:-4] + "_mask.png"
            if os.path.exists(mask):
                pairs.append((p, mask))
    return pairs


def _fit_axis_from_labels(labels: list, axis: str, img_h: int):
    """Fit px<->value from GT label boxes (value = text, px = box centre).

    Matches the evaluation basis: GT CSV data points are compared
    against predictions through this same tick-label mapping, so a
    model trained on masks rebuilt through it extracts curves that
    agree with the GT interpolation (validated: 0.4px mean vs the
    single-curve 512 model on val_single).
    """
    from mci.utils import parse_number_text
    pts = []
    for it in labels:
        box = np.asarray(it["box"], dtype=float)
        c = box.mean(axis=0)
        v = parse_number_text(str(it["text"]))
        if v is None:
            continue
        if axis == "x" and c[1] > img_h - 80:
            pts.append((float(c[0]), v))
        elif axis == "y" and c[0] < 100 and c[1] < img_h - 85:
            pts.append((float(c[1]), v))
    if len(pts) < 2:
        return None
    pts.sort()
    p = np.array([q[0] for q in pts])
    v = np.array([q[1] for q in pts])
    if (v > 0).all() and (float(v.max()) / float(v.min()) > 100):
        a, b = np.polyfit(p, np.log10(v), 1)
        return ("log", float(a), float(b))
    a, b = np.polyfit(p, v, 1)
    return ("linear", float(a), float(b))


def _px_of(fit, v):
    kind, a, b = fit
    vv = np.asarray(v, dtype=np.float64)
    if kind == "log":
        vv = np.where(vv > 0, vv, 1e-9)  # non-positive values: clamp
        return (np.log10(vv) - b) / a
    return (vv - b) / a


def _build_chain_targets(curves_json: dict, base_dir: str, axes: tuple,
                         img_size: tuple, k: int = K, sigma: float = 2.0) -> np.ndarray:
    """Per-column GT curve-position targets (杠杆 3).

    For each curve, map its data points to pixels through the SAME axes
    mapping the evaluator uses, interpolate to every column, and render a
    vertical Gaussian (sigma px) at the curve position into (K, H, W)
    float32.  The chain loss then aligns the model's per-column
    probability centroid with these peaks -- directly optimizing the
    pixel-level curve position that the evaluator measures (kills the
    +0.67px systematic peak offset at the root instead of post-hoc
    calibration).
    """
    w, h = img_size
    out = np.zeros((k, h, w), np.float32)
    if not curves_json or "curves" not in curves_json or axes is None:
        return out
    x_axis, y_axis = axes
    ys = np.arange(h, dtype=np.float64)
    for i, c in enumerate(curves_json["curves"][:k]):
        csv_path = os.path.join(base_dir, c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        data = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        if len(data) < 2:
            continue
        gx = x_axis.value_to_pixel(np.where(data[:, 0] > 0, data[:, 0], 1e-9))
        gy = y_axis.value_to_pixel(np.where(data[:, 1] > 0, data[:, 1], 1e-9))
        o = np.argsort(gx)
        gx, gy = gx[o], gy[o]
        x0, x1 = max(0, int(np.floor(gx[0]))), min(w - 1, int(np.ceil(gx[-1])))
        if x1 <= x0:
            continue
        cols = np.arange(x0, x1 + 1)
        gy_i = np.interp(cols, gx, gy)
        for x, yv in zip(cols, gy_i):
            if not np.isfinite(yv) or yv < 0 or yv >= h:
                continue
            yy = int(round(yv))
            lo, hi = max(0, yy - 8), min(h, yy + 9)
            out[i, lo:hi, x] += np.exp(-0.5 * ((ys[lo:hi] - yv) / sigma) ** 2)
    return out


def _build_chain_targets_x(curves_json: dict, base_dir: str, axes: tuple,
                           img_size: tuple, k: int = K, sigma: float = 2.0) -> np.ndarray:
    """Per-ROW x-position chain targets (杠杆 3 x-direction extension).

    Same idea as _build_chain_targets but along the ROW axis: interpolate
    each curve to every row it spans (y is monotonic in the synthetic
    creep curves) and render a horizontal Gaussian at the x position.
    Aligning the per-row x centroid sharpens the curve position along x
    (steep tails, log-x axes) -- the y-only chain loss leaves x to the
    implicit skeleton/mask constraint.
    """
    w, h = img_size
    out = np.zeros((k, h, w), np.float32)
    if not curves_json or "curves" not in curves_json or axes is None:
        return out
    x_axis, y_axis = axes
    xs = np.arange(w, dtype=np.float64)
    for i, c in enumerate(curves_json["curves"][:k]):
        csv_path = os.path.join(base_dir, c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        data = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        if len(data) < 2:
            continue
        gx = x_axis.value_to_pixel(np.where(data[:, 0] > 0, data[:, 0], 1e-9))
        gy = y_axis.value_to_pixel(np.where(data[:, 1] > 0, data[:, 1], 1e-9))
        o = np.argsort(gy)  # rows are monotonic in y
        gx, gy = gx[o], gy[o]
        y0, y1 = max(0, int(np.floor(gy[0]))), min(h - 1, int(np.ceil(gy[-1])))
        if y1 <= y0:
            continue
        rows_ = np.arange(y0, y1 + 1)
        gx_i = np.interp(rows_, gy, gx)
        for y, xv in zip(rows_, gx_i):
            if not np.isfinite(xv) or xv < 0 or xv >= w:
                continue
            xx = int(round(xv))
            lo, hi = max(0, xx - 8), min(w, xx + 9)
            out[i, y, lo:hi] += np.exp(-0.5 * ((xs[lo:hi] - xv) / sigma) ** 2)
    return out


def chain_loss(logit: torch.Tensor, chain_target: torch.Tensor) -> torch.Tensor:
    """Column-position chain loss (杠杆 3), stable variant.

    v1 (centroid L1): background prob pulls the whole-column centroid -> crash.
    v2 (band BCE): BCE pushes logit -> +inf (unbounded), crashing the model
        (val 0.65 -> 0.12; also seen locally).
    v3 (+exclude overlapping columns): still unstable.
    v4: L2 regression of sigmoid(logit) onto the gaussian band, restricted
        to columns with a SINGLE active band.  The gradient
        2*(p-t)*p*(1-p) vanishes as p->1, so the logit never explodes and
        the loss only sharpens the position; no approach-zone conflict
        (lever 1 owns those columns).
    """
    band = (chain_target > 0.05).float()
    if float(band.sum()) < 1.0:
        return torch.zeros((), device=logit.device)
    col_overlap = (band.sum(dim=1, keepdim=True) > 1.5).float()
    band_eff = band * (1.0 - col_overlap)
    if float(band_eff.sum()) < 1.0:
        return torch.zeros((), device=logit.device)
    prob = torch.sigmoid(logit)
    err = ((prob - chain_target) ** 2) * band_eff
    return err.sum() / band_eff.sum().clamp(min=1.0)


def _meta_instance_masks(meta: dict, labels: list, curves_json: dict,
                        base_dir: str, img_size: tuple, k: int = K,
                        width: int = 3, axes: tuple = None) -> np.ndarray:
    """K-channel instance masks rebuilt from the GT CSVs through the
    tick-label mapping (the evaluation basis).

    Replaces the curves_px polyline masks: those 16 sampled pixels do
    not trace the full rendered curve (160+ data points), so a model
    trained on them drifts from the GT interpolation.  Rebuilding the
    dense curve through the SAME mapping that the evaluator uses makes
    the training target and the acceptance metric consistent.

    ``axes`` = (x_axis, y_axis) built by the SAME code path as the
    evaluator (detect_structure -> read_ticks -> build_axes).  When
    None, falls back to the legacy _fit_axis_from_labels polyfit
    (kept only for backward compatibility / tests).
    """
    w, h = img_size
    out = np.zeros((k, h, w), np.uint8)
    if not curves_json or "curves" not in curves_json:
        return out
    if axes is not None:
        x_axis, y_axis = axes
        for i, c in enumerate(curves_json["curves"][:k]):
            csv_path = os.path.join(base_dir, c["csv"])
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, encoding="utf-8") as f:
                rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
            data = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
            if len(data) < 2:
                continue
            # log-axis NaN guard (same clamp as the legacy _px_of)
            gx = x_axis.value_to_pixel(np.where(data[:, 0] > 0, data[:, 0], 1e-9))
            gy = y_axis.value_to_pixel(np.where(data[:, 1] > 0, data[:, 1], 1e-9))
            pts = np.array([[int(round(x)), int(round(y))] for x, y in zip(gx, gy)],
                          dtype=np.int32)
            cv2.polylines(out[i], [pts], False, 255, thickness=width,
                          lineType=cv2.LINE_AA)
        return out
    fx = _fit_axis_from_labels(labels, "x", h)
    fy = _fit_axis_from_labels(labels, "y", h)
    if fx is None or fy is None:
        return out
    for i, c in enumerate(curves_json["curves"][:k]):
        csv_path = os.path.join(base_dir, c["csv"])
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csvlib.reader(f) if r and not r[0].startswith("#")]
        data = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        if len(data) < 2:
            continue
        gx = _px_of(fx, data[:, 0])
        gy = _px_of(fy, data[:, 1])
        pts = np.array([[int(round(x)), int(round(y))] for x, y in zip(gx, gy)],
                      dtype=np.int32)
        cv2.polylines(out[i], [pts], False, 255, thickness=width,
                      lineType=cv2.LINE_AA)
    return out




class ChartDataset(Dataset):
    def __init__(self, pairs, augment=False, size=SIZE, cache=None, skel_cache=None,
                 chain_cache=None, chainx_cache=None):
        self.pairs = pairs
        self.augment = augment
        self.size = size
        self.cache = cache  # dict path -> K-channel mask (for single-curve reuse)
        self.skel_cache = skel_cache  # dict path -> K-channel GT skeleton
        self.chain_cache = chain_cache  # dict path -> (K,H,W) chain targets
        self.chainx_cache = chainx_cache  # dict path -> (K,H,W) row-wise x targets

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        sem = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or sem is None:
            raise FileNotFoundError(img_path)
        meta_p = img_path[:-4] + "_meta.json"
        meta = {}
        if os.path.exists(meta_p):
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
        key = img_path
        chain = None
        chain_x = None
        if self.cache is not None and key in self.cache:
            inst = self.cache[key]
            skel = (self.skel_cache or {}).get(key)
            if skel is None:
                skel = _instance_skeletons(inst)
                if self.skel_cache is not None:
                    self.skel_cache[key] = skel
            if self.chain_cache is not None and key in self.chain_cache:
                chain = self.chain_cache[key]
            if self.chainx_cache is not None and key in self.chainx_cache:
                chain_x = self.chainx_cache[key]
        else:
            labels = []
            lab_p = img_path[:-4] + "_labels.json"
            if os.path.exists(lab_p):
                with open(lab_p, encoding="utf-8") as f:
                    labels = json.load(f)
            curves_json = {}
            cj_p = img_path[:-4] + "_curves.json"
            if os.path.exists(cj_p):
                with open(cj_p, encoding="utf-8") as f:
                    curves_json = json.load(f)
            # Phase C fix: build axes with the SAME code path the
            # evaluator uses (detect_structure -> read_ticks ->
            # build_axes), so the training mask target is pixel-
            # consistent with the evaluation basis.  The legacy
            # _fit_axis_from_labels polyfit mis-judged log axes on
            # ~25% of training images (bottom y tick label excluded
            # by the img_h-85 filter -> 0.1/1/10 seen as linear),
            # teaching the model a systematically wrong target.
            axes = None
            try:
                from mci.pipeline.chart_structure import detect_structure
                from mci.pipeline.coordinate_mapper import build_axes
                from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
                from mci.pipeline.extractor import load_config as _lc
                from mci.utils import read_image as _ri
                cfg = _lc()
                img_bgr = _ri(img_path)
                structure = detect_structure(img_bgr, cfg)
                ocr = StubOCRBackend(lab_p)
                x_ticks, y_ticks = read_ticks(img_bgr, structure, ocr, cfg)
                x_axis, y_axis = build_axes(
                    x_ticks, y_ticks,
                    x_endpoints=(float(structure.y_axis_pixel), float(structure.plot_bbox[2])),
                    y_endpoints=(float(structure.plot_bbox[1]), float(structure.x_axis_pixel)),
                )
                axes = (x_axis, y_axis)
            except Exception:
                axes = None  # fall back to legacy fit below
            inst = _meta_instance_masks(meta, labels, curves_json,
                                        os.path.dirname(img_path),
                                        (img.shape[1], img.shape[0]), axes=axes)
            skel = _instance_skeletons(inst)
            if self.chain_cache is not None and axes is not None:
                chain = _build_chain_targets(curves_json, os.path.dirname(img_path),
                                             axes, (img.shape[1], img.shape[0]))
                # cache at TRAINING resolution (512): full-res float32
                # chain targets for 3300 images blew the container memory
                # (121 GB > cgroup limit, silent kill)
                chain = cv2.resize(chain.transpose(1, 2, 0), (self.size, self.size),
                                   interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                self.chain_cache[key] = chain
            if self.chainx_cache is not None and axes is not None:
                chain_x = _build_chain_targets_x(curves_json, os.path.dirname(img_path),
                                                 axes, (img.shape[1], img.shape[0]))
                chain_x = cv2.resize(chain_x.transpose(1, 2, 0), (self.size, self.size),
                                     interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                self.chainx_cache[key] = chain_x
            if self.cache is not None:
                # cache at training resolution too (saves ~11 GB for 3300
                # images vs full-res uint8 masks; training always resizes)
                inst512 = cv2.resize(inst.transpose(1, 2, 0), (self.size, self.size),
                                     interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
                self.cache[key] = inst512
                inst = inst512
                if self.skel_cache is not None:
                    skel512 = cv2.resize(skel.transpose(1, 2, 0), (self.size, self.size),
                                         interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
                    self.skel_cache[key] = skel512
                    skel = skel512
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        inst = np.ascontiguousarray(inst)
        skel = np.ascontiguousarray(skel)
        if chain is not None:
            # already cached at training resolution (512)
            chain = np.ascontiguousarray(chain)
        if chain_x is not None:
            chain_x = np.ascontiguousarray(chain_x)

        if self.augment:
            img, inst, skel, chain, chain_x = _augment(img, inst, skel, chain, chain_x)

        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        y = (torch.from_numpy(inst).float() / 255.0 > 0.5).float()
        s = (torch.from_numpy(skel).float() / 255.0 > 0.5).float()
        c = torch.from_numpy(chain).float() if chain is not None else None
        cx = torch.from_numpy(chain_x).float() if chain_x is not None else None
        return x, y, s, c, cx


def _instance_skeletons(inst: np.ndarray) -> np.ndarray:
    """Per-channel binary skeleton of the instance masks (K,H,W uint8).

    Used as the GT side of the skeleton-recall loss (clDice-family):
    the loss only needs the GT skeleton (precomputed once per image),
    and penalizes predictions that miss the skeleton pixels -- directly
    targeting dash/gap/jump failures on thin structures.
    """
    from skimage.morphology import skeletonize
    k, h, w = inst.shape
    out = np.zeros((k, h, w), np.uint8)
    for c in range(k):
        m = inst[c] > 0
        if m.sum() < 4:
            continue
        out[c] = (skeletonize(m).astype(np.uint8)) * 255
    return out


def _augment(img: np.ndarray, inst: np.ndarray, skel: np.ndarray = None,
             chain: np.ndarray = None, chain_x: np.ndarray = None) -> tuple:
    """Augment image + K-channel masks (and skeletons, chain targets)."""
    size = img.shape[0]
    if random.random() < 0.5:
        # Horizontal mirror (left-right) of image + K-channel masks.
        # NOTE: cv2.flip(inst, 2) on a (K,H,W) array is WRONG: at size 512
        # OpenCV's python binding treats it as a multi-channel image and
        # flips the HEIGHT axis (vertical mirror, channels kept), so image
        # and masks were misaligned; at size 768 it crashes outright
        # (dims>2 assert, channel heuristic cutoff = 512).  Use numpy:
        # mirror the width axis AND reverse the channel order (channels are
        # sorted by curve x-start, which reverses under a horizontal flip).
        img = cv2.flip(img, 1)
        inst = np.flip(inst, axis=2)[::-1].copy()
        if skel is not None:
            skel = np.flip(skel, axis=2)[::-1].copy()
        if chain is not None:
            chain = np.flip(chain, axis=2)[::-1].copy()
        if chain_x is not None:
            chain_x = np.flip(chain_x, axis=2)[::-1].copy()
    if random.random() < 0.4:
        g = random.uniform(0.8, 1.25)
        b = random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        ang = random.uniform(-3, 3)
        m = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        img = cv2.warpAffine(img, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=255)
        for c in range(inst.shape[0]):
            inst[c] = cv2.warpAffine(inst[c], m, (size, size),
                                     flags=cv2.INTER_NEAREST, borderValue=0)
            if skel is not None:
                skel[c] = cv2.warpAffine(skel[c], m, (size, size),
                                         flags=cv2.INTER_NEAREST, borderValue=0)
            if chain is not None:
                chain[c] = cv2.warpAffine(chain[c], m, (size, size),
                                          flags=cv2.INTER_LINEAR, borderValue=0)
            if chain_x is not None:
                chain_x[c] = cv2.warpAffine(chain_x[c], m, (size, size),
                                            flags=cv2.INTER_LINEAR, borderValue=0)
    if random.random() < 0.25:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 90)])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if random.random() < 0.3:
        f = random.uniform(0.82, 1.0)
        cs = int(size * f)
        x0 = random.randint(0, size - cs)
        y0 = random.randint(0, size - cs)
        img = cv2.resize(img[y0:y0 + cs, x0:x0 + cs], (size, size), interpolation=cv2.INTER_LINEAR)
        inst_c = inst[:, y0:y0 + cs, x0:x0 + cs]
        inst = np.stack([cv2.resize(inst_c[c], (size, size),
                                    interpolation=cv2.INTER_NEAREST)
                         for c in range(inst.shape[0])])
        if skel is not None:
            skel_c = skel[:, y0:y0 + cs, x0:x0 + cs]
            skel = np.stack([cv2.resize(skel_c[c], (size, size),
                                        interpolation=cv2.INTER_NEAREST)
                             for c in range(skel.shape[0])])
        if chain is not None:
            chain_c = chain[:, y0:y0 + cs, x0:x0 + cs]
            chain = np.stack([cv2.resize(chain_c[c], (size, size),
                                         interpolation=cv2.INTER_LINEAR)
                              for c in range(chain.shape[0])])
        if chain_x is not None:
            chain_x_c = chain_x[:, y0:y0 + cs, x0:x0 + cs]
            chain_x = np.stack([cv2.resize(chain_x_c[c], (size, size),
                                           interpolation=cv2.INTER_LINEAR)
                                for c in range(chain_x.shape[0])])
    if random.random() < 0.15:
        noise = (np.random.default_rng().random(img.shape) < 0.0008)
        img = img.copy()
        img[noise] = 0
        img[np.random.default_rng().random(img.shape) < 0.0008] = 255
    return img, inst, skel, chain, chain_x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/train_platform,data/train_platform_single")
    ap.add_argument("--val-dir", default="data/val_multi,data/val_single")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/checkpoints/unet_multi_curve.pt")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--init", default=None,
                    help="pretrained single-curve UNet (encoder weights shared)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-dir-limit", type=int, default=0,
                    help="cap training pairs PER directory (balanced subset)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay for weight averaging (0 disables)")
    ap.add_argument("--base", type=int, default=64,
                    help="UNet base channel count (方案 H: 96/128 need GPU > 8GB)")
    ap.add_argument("--embed-dim", type=int, default=16,
                    help="GOI embedding dimension (方案 E; 0 disables the head)")
    ap.add_argument("--goi-weight", type=float, default=0.1,
                    help="GOI intra-class pull loss weight")
    ap.add_argument("--goi-ortho", type=float, default=0.1,
                    help="GOI inter-class orthogonality loss weight")
    ap.add_argument("--zone-dist", type=int, default=4,
                    help="杠杆1: approach-zone dilation distance (px)")
    ap.add_argument("--zone-margin", type=float, default=0.2,
                    help="杠杆1: contrastive hinge margin (cosine)")
    ap.add_argument("--zone-contrast", type=float, default=0.5,
                    help="杠杆1: approach-zone contrastive loss weight (0 disables)")
    ap.add_argument("--chain-weight", type=float, default=0.0,
                    help="杠杆3: chain loss weight - align column probability "
                         "centroids with GT curve positions (0 disables)")
    ap.add_argument("--right-weight", type=float, default=1.0,
                    help="E1: right-end mask loss multiplier (x_frac>0.72; "
                         "1.0 disables). Targets right-end curve loss "
                         "(chain-model diagnosis: ~50% of >5% failures).")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    use_amp = device.startswith("cuda")
    print(f"device: {device}, amp: {use_amp}")

    pairs = _list_pairs(args.data_dir)
    val_pairs = _list_pairs(args.val_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if args.per_dir_limit > 0:
        per = []
        for p, m in pairs:
            d = os.path.dirname(p)
            cnt = sum(1 for q, _ in per if os.path.dirname(q) == d)
            if cnt < args.per_dir_limit:
                per.append((p, m))
        pairs = per
    print(f"train pairs: {len(pairs)}, val pairs: {len(val_pairs)}")
    if len(pairs) < 10:
        print("ERROR: too few training images")
        return 1

    cache: dict = {}
    skel_cache: dict = {}
    chain_cache: dict = {}
    chainx_cache: dict = {}
    train_ds = ChartDataset(pairs, augment=True, size=args.size, cache=cache,
                            skel_cache=skel_cache, chain_cache=chain_cache,
                            chainx_cache=chainx_cache)
    val_ds = ChartDataset(val_pairs, augment=False, size=args.size, cache=cache,
                          skel_cache=skel_cache, chain_cache=chain_cache,
                          chainx_cache=chainx_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(in_channels=1, base=args.base, out_channels=K,
                 embed_dim=args.embed_dim).to(device)
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        # encoder weights transfer; the output head (1 -> K channels) is new
        # when init is a single-curve model, but when init is a K-channel
        # multi model (e.g. continuing 512c) the out head must transfer too.
        # NOTE: base 96/128 init from base-64 checkpoints only transfers the
        # shallow layers whose channel counts match (enc1/enc2); deeper ones
        # are randomly initialized (printed as missing).
        # embed_head: transferred when the init checkpoint has one (方案 E
        # continuation); otherwise randomly initialized (printed as missing).
        sd = {k: v for k, v in sd.items() if k.startswith("enc") or k.startswith("bottleneck")
              or k.startswith("up") or k.startswith("dec") or k.startswith("out")
              or k.startswith("embed")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"initialized from {args.init} (missing={len(missing)} unexpected={len(unexpected)})")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # EMA (zero-risk stabilizer; STU-Net/SegFormer-style evaluation uses the
    # averaged weights -- keeps the thin-structure precision improvements)
    ema_decay = float(getattr(args, "ema_decay", 0.999))
    ema_model = None
    if ema_decay > 0:
        ema_model = UNet(in_channels=1, base=args.base, out_channels=K,
                         embed_dim=args.embed_dim).to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()

    def _ema_update():
        if ema_model is None:
            return
        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema_model.parameters()):
                ep.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
            for b, eb in zip(model.buffers(), ema_model.buffers()):
                eb.copy_(b)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, n = 0.0, 0
        for x, y, s, c, cx in train_loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            chain_batch = None
            chainx_batch = None
            if args.chain_weight > 0:
                chain_batch = torch.stack([
                    (ci.to(device) if ci is not None else torch.zeros_like(s[0]))
                    for ci in c])
                chainx_batch = torch.stack([
                    (ci.to(device) if ci is not None else torch.zeros_like(s[0]))
                    for ci in cx])
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                # channel weights: empty channels (target all-zero) are
                # down-weighted so the model does not learn to leave the
                # higher curve channels empty (4/5-curve charts are rarer
                # in the training distribution)
                w = (y.sum(dim=(2, 3)) > 0).float() * 0.7 + 0.3
                # E1: right-end (x_frac > 0.72) pixel-weight multiplier --
                # the chain model loses curves in the right half under
                # heavy blur/low-quality screenshots (right-end drift).
                # Weighting the GT mask pixels there pushes the model to
                # keep the chain at the exact position in low-confidence
                # zones.  Passed as per-pixel BCE weight (B, 1, 1, W).
                px_w = None
                if args.right_weight > 1.0:
                    xs = torch.arange(y.shape[3], device=y.device).float()
                    right = (xs / float(y.shape[3] - 1) > 0.72).float()
                    px_w = 1.0 + (args.right_weight - 1.0) * right[None, None, None, :]
                logit, emb = model.forward_embed(x)
                loss = (bce_dice_loss(logit, y, weight=px_w) * w).mean()
                # skeleton-recall term (clDice family): penalize missing the
                # GT skeleton pixels -- directly targets dash/gap/jump
                # failures on thin structures; only needs the GT skeleton.
                if s.sum() > 0:
                    prob = torch.sigmoid(logit)
                    num = (prob * s).sum(dim=(1, 2, 3))
                    den = s.sum(dim=(1, 2, 3)) + 1e-6
                    skel_recall = (num / den).mean()
                    loss = loss + 0.5 * (1.0 - skel_recall)
                # 杠杆 3: chain loss -- gaussian-band weighted L2 pulling
                # probability mass to the exact GT curve position (kills the
                # +0.67px systematic peak offset at the root).  The x
                # direction uses the transposed (per-row) targets.
                if chain_batch is not None:
                    loss = loss + args.chain_weight * chain_loss(logit, chain_batch)
                    if chainx_batch is not None:
                        loss = loss + args.chain_weight * chain_loss(
                            logit.transpose(2, 3), chainx_batch.transpose(2, 3))
                # GOI terms (方案 E): intra-class pull + inter-class global
                # orthogonality on the embedding head (0 when disabled).
                # 杠杆 1: approach-zone contrastive loss when embed_dim > 0.
                if args.embed_dim > 0 and (y > 0).any():
                    near = approach_zone_mask(y, dist=args.zone_dist)
                    if args.zone_contrast > 0 and bool(near.any()):
                        pull, ortho, contr = goi_loss_contrastive(
                            emb, y, near, margin=args.zone_margin)
                        loss = (loss + args.goi_weight * pull
                                + args.goi_ortho * ortho
                                + args.zone_contrast * contr)
                    else:
                        pull, ortho = goi_loss(emb, y)
                        loss = (loss + args.goi_weight * pull
                                + args.goi_ortho * ortho)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            _ema_update()
            tot_loss += loss.item() * len(x)
            n += len(x)
        sched.step()

        model.eval()
        v_dice = v_iou = 0.0
        if len(val_loader):
            with torch.no_grad():
                for x, y, s, c, cx in val_loader:
                    x, y = x.to(device), y.to(device)
                    prob = torch.sigmoid((ema_model or model)(x))
                    d, i = dice_iou(prob, y)
                    v_dice += d * len(x)
                    v_iou += i * len(x)
            v_dice /= len(val_ds)
            v_iou /= len(val_ds)
        else:
            v_dice = v_iou = float("nan")

        print(f"[{epoch:02d}/{args.epochs}] loss={tot_loss / max(n, 1):.4f} "
              f"val_dice={v_dice:.4f} val_iou={v_iou:.4f} "
              f"lr={sched.get_last_lr()[0]:.2e} ({time.time() - t0:.1f}s)")

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save({"state_dict": (ema_model or model).state_dict(),
                        "epoch": epoch, "val_iou": v_iou, "val_dice": v_dice,
                        "ema_decay": ema_decay, "embed_dim": args.embed_dim,
                        "base": args.base, "size": args.size}, args.out)
            print(f"  -> saved {args.out} (val_iou={v_iou:.4f}, ema)")

    print(f"done. best val_iou = {best_iou:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
