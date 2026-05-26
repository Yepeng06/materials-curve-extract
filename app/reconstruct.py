from __future__ import annotations

import numpy as np

from app.config import PlotArea


def _resample_points(points: list[tuple[float, float]], resample_n: int) -> list[tuple[float, float]]:
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    target_xs = np.linspace(xs.min(), xs.max(), resample_n)
    target_ys = np.interp(target_xs, xs, ys)
    return list(zip(target_xs.tolist(), target_ys.tolist()))


def mask_to_curve_points(mask: np.ndarray, plot_area: PlotArea, resample_n: int | None = None) -> list[tuple[float, float]]:
    if mask.size == 0 or np.count_nonzero(mask) == 0:
        raise ValueError("未检测到曲线像素，请调整 plot_area、mode 或阈值参数")

    local_ys, local_xs = np.where(mask > 0)
    if len(local_xs) == 0:
        raise ValueError("未检测到曲线像素，请调整 plot_area、mode 或阈值参数")

    points_local: list[tuple[float, float]] = []
    for x in np.unique(local_xs):
        ys = local_ys[local_xs == x]
        y_mid = float(np.median(ys))
        points_local.append((float(x), y_mid))

    points_local.sort(key=lambda p: p[0])

    if len(points_local) < 5:
        raise ValueError("检测到的曲线点过少（<5），请调整 plot_area、mode 或阈值参数")

    if resample_n is not None and len(points_local) > resample_n:
        points_local = _resample_points(points_local, resample_n)

    return [(x + plot_area.left, y + plot_area.top) for x, y in points_local]
