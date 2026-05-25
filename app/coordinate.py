from __future__ import annotations

import numpy as np

from app.config import AxisRange


def pixel_to_data(points_xy: np.ndarray, width: int, height: int, x_range: AxisRange, y_range: AxisRange) -> np.ndarray:
    """Map pixel coordinates (x rightward, y downward) to data coordinates."""
    if width <= 1 or height <= 1:
        raise ValueError("width and height must be > 1")

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_data = x_range.min + (x / (width - 1)) * (x_range.max - x_range.min)
    y_data = y_range.max - (y / (height - 1)) * (y_range.max - y_range.min)
    return np.column_stack([x_data, y_data])
