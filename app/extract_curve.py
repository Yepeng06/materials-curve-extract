from __future__ import annotations

import numpy as np


def extract_curve_mask(cropped: np.ndarray, mode: str = "gray") -> np.ndarray:
    """V0 placeholder for curve extraction."""
    h, w = cropped.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    return mask
