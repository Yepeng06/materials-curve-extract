"""Visualization: overlay extraction result on the source image."""
from __future__ import annotations

import os

import cv2
import numpy as np

from ..schema import ExtractionResult


def overlay_result(result: ExtractionResult, image_bgr: np.ndarray, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dbg = image_bgr.copy()
    x0, y0, x1, y1 = result.structure.plot_bbox
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
    for px in result.structure.x_ticks_px:
        cv2.drawMarker(dbg, (int(px), result.structure.y_axis_pixel), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    for py in result.structure.y_ticks_px:
        cv2.drawMarker(dbg, (result.structure.x_axis_pixel, int(py)), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    for c in result.curves:
        for (px, py) in c.pixel_points[::3]:
            cv2.circle(dbg, (px, py), 2, (0, 0, 255), -1)
    cv2.imwrite(out_path, dbg)
    return out_path
