from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import PlotArea


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def crop_plot_area(image: np.ndarray, plot_area: PlotArea) -> np.ndarray:
    return image[plot_area.top : plot_area.bottom, plot_area.left : plot_area.right]
