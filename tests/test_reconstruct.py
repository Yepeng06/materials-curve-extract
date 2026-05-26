import cv2
import numpy as np

from app.config import PlotArea
from app.reconstruct import mask_to_curve_points


def test_mask_to_curve_points_sorted_and_global_coords():
    mask = np.zeros((80, 120), dtype=np.uint8)
    cv2.line(mask, (5, 60), (110, 10), 255, 2)
    plot_area = PlotArea(left=100, top=200, right=220, bottom=280)

    points = mask_to_curve_points(mask, plot_area)

    assert len(points) > 0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert xs == sorted(xs)
    assert min(xs) >= plot_area.left
    assert max(xs) < plot_area.right
    assert min(ys) >= plot_area.top
    assert max(ys) < plot_area.bottom
