from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np

from app.config import AxisRange, PlotArea


@dataclass(frozen=True)
class MappedPoint:
    pixel_x: float
    pixel_y: float
    x: float
    y: float


def _validate_ranges(plot_area: PlotArea, x_range: AxisRange, y_range: AxisRange) -> tuple[float, float]:
    width = plot_area.right - plot_area.left
    height = plot_area.bottom - plot_area.top
    if width <= 0 or height <= 0:
        raise ValueError("plot_area width and height must be > 0")
    if x_range.max <= x_range.min:
        raise ValueError("x_range max must be greater than min")
    if y_range.max <= y_range.min:
        raise ValueError("y_range max must be greater than min")
    return float(width), float(height)


def pixel_to_data_point(
    pixel_x: float,
    pixel_y: float,
    plot_area: PlotArea,
    x_range: AxisRange,
    y_range: AxisRange,
) -> dict:
    width, height = _validate_ranges(plot_area, x_range, y_range)

    x = x_range.min + (float(pixel_x) - plot_area.left) / width * (x_range.max - x_range.min)
    y = y_range.max - (float(pixel_y) - plot_area.top) / height * (y_range.max - y_range.min)
    return asdict(MappedPoint(pixel_x=float(pixel_x), pixel_y=float(pixel_y), x=float(x), y=float(y)))


def pixel_to_data_points(
    pixel_points: Sequence[Sequence[float]] | np.ndarray,
    plot_area: PlotArea,
    x_range: AxisRange,
    y_range: AxisRange,
) -> list[dict]:
    _validate_ranges(plot_area, x_range, y_range)
    if pixel_points is None:
        raise ValueError("pixel_points must not be None")

    if isinstance(pixel_points, np.ndarray):
        if pixel_points.size == 0:
            return []
        if pixel_points.ndim != 2 or pixel_points.shape[1] != 2:
            raise ValueError("pixel_points numpy array must have shape (N, 2)")
        pairs: Iterable[Sequence[float]] = pixel_points.tolist()
    else:
        if len(pixel_points) == 0:
            return []
        pairs = pixel_points

    out: list[dict] = []
    for idx, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise ValueError(f"pixel point at index {idx} must be a 2-item sequence")
        try:
            px = float(pair[0])
            py = float(pair[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pixel point at index {idx} contains non-numeric values") from exc
        out.append(pixel_to_data_point(px, py, plot_area, x_range, y_range))
    return out
