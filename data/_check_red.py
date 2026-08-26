# -*- coding: utf-8 -*-
"""Analyze web-output images in temp: detect red overlay (extracted curves)."""
import os, glob
import numpy as np
import cv2

for fp in sorted(glob.glob(r"F:\CODE\New\temp\*.png")):
    img = cv2.imread(fp)
    if img is None:
        print(os.path.basename(fp), "unreadable"); continue
    h, w = img.shape[:2]
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    # red overlay points: high R, low G/B
    red = (r > 150) & (g < 100) & (b < 100)
    red_frac = red.mean()
    # dark pixels (ink / curves / text)
    dark = (r < 100) & (g < 100) & (b < 100)
    dark_frac = dark.mean()
    # red points that lie ON dark ink (red drawn over dark curve) vs on white
    red_on_dark = (red & dark).sum() / max(1, red.sum())
    print(f"{os.path.basename(fp)[:20]:22s} {w}x{h} red={red_frac:.4f} dark={dark_frac:.4f} red_on_dark={red_on_dark:.2f}")
