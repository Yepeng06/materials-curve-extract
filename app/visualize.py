from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_mask(mask: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), mask)


def save_overlay(base: np.ndarray, mask: np.ndarray, path: Path) -> None:
    overlay = base.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = (0, 0, 255)
    cv2.imwrite(str(path), overlay)


def save_redrawn(mask: np.ndarray, path: Path) -> None:
    h, w = mask.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    canvas[ys, xs] = (0, 0, 0)
    cv2.imwrite(str(path), canvas)
