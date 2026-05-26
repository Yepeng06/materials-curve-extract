from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def extract_curve_mask(
    cropped_image: np.ndarray,
    mode: str = "gray",
    hsv_lower: tuple[int, int, int] | None = None,
    hsv_upper: tuple[int, int, int] | None = None,
    options: dict[str, Any] | None = None,
) -> np.ndarray:
    opts = options or {}
    blur_ksize = int(opts.get("blur_ksize", 5))
    morph_kernel = int(opts.get("morph_kernel", 3))
    min_area = int(opts.get("min_area", 20))

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    if morph_kernel < 1:
        morph_kernel = 1

    if mode == "gray":
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif mode == "hsv":
        if hsv_lower is None or hsv_upper is None:
            raise ValueError("hsv 模式必须提供 hsv_lower 与 hsv_upper")
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))
    else:
        raise ValueError(f"不支持的 mode: {mode}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = remove_small_components(mask, min_area=min_area)

    out = np.where(mask > 0, 255, 0).astype(np.uint8)
    return out
