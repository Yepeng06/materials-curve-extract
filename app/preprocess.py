from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import PlotArea


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image


def crop_plot_area(image: np.ndarray, plot_area: PlotArea) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = plot_area.left, plot_area.top, plot_area.right, plot_area.bottom

    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"plot_area 超出图像范围: image_size=({width},{height}), "
            f"plot_area=({left},{top},{right},{bottom})"
        )
    if right <= left or bottom <= top:
        raise ValueError("plot_area 非法: 需要满足 right>left 且 bottom>top")

    return image[top:bottom, left:right]


def save_image(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise ValueError(f"保存图像失败: {output_path}")
