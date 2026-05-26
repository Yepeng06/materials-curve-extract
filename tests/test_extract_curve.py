import cv2
import numpy as np

from app.extract_curve import extract_curve_mask


def test_extract_curve_mask_gray_non_empty():
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    pts = np.array([[20, 180], [70, 140], [120, 120], [180, 80], [260, 40]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 0), 2)

    mask = extract_curve_mask(img, mode="gray")

    assert mask.dtype == np.uint8
    assert mask.shape[:2] == img.shape[:2]
    assert np.count_nonzero(mask == 255) > 0
