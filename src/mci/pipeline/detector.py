"""YOLOv8-nano structure detection (Phase B-4): detection -> ChartStructure.

Wraps an ultralytics YOLO model trained on the 6-class structure set
(plot_area, x_axis_line, y_axis_line, tick_label, legend_box, axis_title)
and converts its boxes into the same ChartStructure the CV detector
returns, so the rest of the pipeline is untouched.

  plot_area   -> plot_bbox
  x_axis_line -> x_axis_pixel (y centre of the bottom axis box)
  y_axis_line -> y_axis_pixel (x centre of the left axis box)
  tick_label  -> x_ticks_px / y_ticks_px (box centres)
  axis_title  -> kept in structure.meta for the title reader

Missing x/y axis-line detections are derived from plot_area edges
(the bottom edge row and the left edge column), which are near-perfect.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schema import ChartStructure, StructureDetectionError

YOLO_NAMES = {
    0: "plot_area", 1: "x_axis_line", 2: "y_axis_line",
    3: "tick_label", 4: "legend_box", 5: "axis_title",
}


class YoloStructureDetector:
    """Detector backend producing the standard ChartStructure."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise StructureDetectionError(
                "ultralytics not installed; use structure_backend: cv"
            ) from e
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image_bgr: np.ndarray) -> ChartStructure:
        """Run inference and map boxes to a ChartStructure (y-down coords)."""
        h, w = image_bgr.shape[:2]
        res = self.model.predict(
            image_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            raise StructureDetectionError("YOLO found no structure boxes")
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        plot_boxes = [b for b, c in zip(xyxy, cls) if c == 0]
        x_axis_boxes = [b for b, c in zip(xyxy, cls) if c == 1]
        y_axis_boxes = [b for b, c in zip(xyxy, cls) if c == 2]
        tick_boxes = [(b, cf) for b, c, cf in zip(xyxy, cls, confs) if c == 3]
        title_boxes = [b for b, c in zip(xyxy, cls) if c == 5]

        if not plot_boxes:
            raise StructureDetectionError("YOLO found no plot_area")
        # largest plot box wins
        pa = max(plot_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x0, y0, x1, y1 = (float(v) for v in pa)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)

        # axis lines: prefer detections, fall back to plot edges
        if x_axis_boxes:
            x_axis_row = float(np.median([b[3] for b in x_axis_boxes]))
        else:
            x_axis_row = y1  # bottom edge of the plot area
        if y_axis_boxes:
            y_axis_col = float(np.median([b[0] for b in y_axis_boxes]))
        else:
            y_axis_col = x0  # left edge of the plot area

        # tick labels -> tick positions (label box centres along the axes)
        x_ticks, y_ticks = [], []
        for (bx0, by0, bx1, by1), cf in tick_boxes:
            cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
            if abs(cy - x_axis_row) < abs(cx - y_axis_col):
                x_ticks.append(cx)
            else:
                y_ticks.append(cy)
        x_ticks = sorted(set(round(v, 1) for v in x_ticks))
        y_ticks = sorted(set(round(v, 1) for v in y_ticks))

        structure = ChartStructure(
            plot_bbox=(int(round(x0)), int(round(y0)),
                       int(round(x1)), int(round(y1))),
            x_axis_pixel=int(round(x_axis_row)),
            y_axis_pixel=int(round(y_axis_col)),
            x_ticks_px=x_ticks,
            y_ticks_px=y_ticks,
        )
        structure.meta = {
            "backend": "yolo",
            "axis_title_boxes": [list(map(float, b)) for b in title_boxes],
        }
        return structure

