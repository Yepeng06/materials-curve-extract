from __future__ import annotations

import numpy as np


def reconstruct_curve_points(mask: np.ndarray, resample_n: int = 512) -> np.ndarray:
    """V0 placeholder for point ordering/re-sampling."""
    xs = np.linspace(0, max(mask.shape[1] - 1, 1), num=min(resample_n, max(mask.shape[1], 2)))
    ys = np.zeros_like(xs)
    return np.column_stack([xs, ys])
