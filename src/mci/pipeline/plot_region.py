"""Plot-region detection (Level-0 structure front-end).

Goal: replace the brittle full-image geometric projection for messy real
material-curve figures (embedded performance tables, dense annotation text,
captions) with a learned region detector that (a) finds the data-ink plot
area and (b) marks non-chart regions to EXCLUDE before the existing CV/YOLO
structure detector and curve pipeline run.

Agreed best approach (see 参考/方案_曲线提取结构检测组件化检测.md):

  Level 0  plot-region + exclusion detection   -> where to look
  Level 1  chart-element detection (in region) -> axes / ticks / legend
  Level 2  existing precision layer            -> RANSAC / log / OCR / U-Net

The module is fully backward compatible: when no detector is configured or
the detector is low-confidence / unavailable, callers fall back to the
unchanged classical-CV structure detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..schema import ChartStructure, StructureDetectionError
from .chart_structure import detect_structure

# Region classes for the plot-region detector.  Class 0 is the data canvas;
# EXCLUSION_CLASSES are non-chart regions masked out so their grid lines /
# text cannot be mistaken for axes / tick labels.
REGION_NAMES = {
    0: "plot_area",
    1: "embedded_table",
    2: "annotation_text",
    3: "caption",
    4: "legend",
    5: "colorbar",
    6: "axis_title",
}
EXCLUSION_CLASSES = {"embedded_table", "annotation_text", "caption"}


@dataclass
class PlotRegion:
    """Detected data-ink region in full-image coordinates."""

    plot_bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    exclusions: List[Tuple[str, Tuple[float, float, float, float]]] = field(
        default_factory=list
    )
    confidence: float = 1.0


class PlotRegionDetector:
    """Protocol: ``detect(image_bgr) -> Optional[PlotRegion]``.

    Returning ``None`` (or a low-confidence region that the caller rejects)
    signals the caller to fall back to classical-CV structure detection.
    """

    def detect(self, image_bgr: np.ndarray) -> Optional[PlotRegion]:
        raise NotImplementedError


class YoloPlotRegionDetector:
    """Ultralytics YOLO backend for the 7 region classes above.

    Loads lazily; if ultralytics is missing or the weights file does not exist
    the instance reports ``available=False`` and ``detect`` returns ``None``,
    so the rest of the pipeline keeps working unchanged.
    """

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self._model = None
        self._available = False
        try:
            import os

            from ultralytics import YOLO

            if os.path.exists(weights):
                self._model = YOLO(weights)
                self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image_bgr: np.ndarray) -> Optional[PlotRegion]:
        if not self._available or self._model is None:
            return None
        h, w = image_bgr.shape[:2]
        res = self._model.predict(
            image_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        plot_boxes = [b for b, c in zip(xyxy, cls) if c == 0]
        if not plot_boxes:
            return None
        pa = max(plot_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x0, y0, x1, y1 = (float(v) for v in pa)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(w - 1.0, x1), min(h - 1.0, y1)

        exclusions: List[Tuple[str, Tuple[float, float, float, float]]] = []
        plot_conf = float(np.max(confs[cls == 0])) if len(plot_boxes) else 0.0
        for b, c, cf in zip(xyxy, cls, confs):
            name = REGION_NAMES.get(int(c))
            if name in EXCLUSION_CLASSES:
                exclusions.append(
                    (name, (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(cf))
                )
        return PlotRegion(
            plot_bbox=(
                int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
            ),
            exclusions=exclusions,
            confidence=plot_conf,
        )


def apply_region(
    image_bgr: np.ndarray, region: PlotRegion
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """Crop to ``plot_area`` and build an exclusion mask in crop coordinates.

    Returns ``(crop, offset_x, offset_y, mask)`` where ``mask`` is a bool
    HxW array (same size as ``crop``) with True inside excluded regions.
    """
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = region.plot_bbox
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w - 1, int(round(x1))), min(h - 1, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        raise StructureDetectionError("invalid plot region bbox")
    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1].copy()
    hc, wc = crop.shape[:2]
    mask = np.zeros((hc, wc), dtype=bool)
    for _name, (bx0, by0, bx1, by1) in region.exclusions:
        cx0 = int(np.clip(bx0 - x0, 0, wc - 1))
        cy0 = int(np.clip(by0 - y0, 0, hc - 1))
        cx1 = int(np.clip(bx1 - x0, 0, wc - 1))
        cy1 = int(np.clip(by1 - y0, 0, hc - 1))
        if cx1 > cx0 and cy1 > cy0:
            mask[cy0 : cy1 + 1, cx0 : cx1 + 1] = True
    return crop, x0, y0, mask


def _offset_structure(struct: ChartStructure, ox: int, oy: int) -> ChartStructure:
    """Map a structure detected inside a crop back to full-image coordinates.

    Conventions (see schema.ChartStructure):
      x_axis_pixel is a ROW (y) -> offset by oy
      y_axis_pixel is a COLUMN (x) -> offset by ox
      x_ticks_px are columns (x) -> +ox
      y_ticks_px are rows (y) -> +oy
    """
    x0, y0, x1, y1 = struct.plot_bbox
    return ChartStructure(
        plot_bbox=(x0 + ox, y0 + oy, x1 + ox, y1 + oy),
        x_axis_pixel=struct.x_axis_pixel + oy,
        y_axis_pixel=struct.y_axis_pixel + ox,
        x_ticks_px=[t + ox for t in struct.x_ticks_px],
        y_ticks_px=[t + oy for t in struct.y_ticks_px],
        meta=dict(struct.meta),
        panels=list(struct.panels),
    )


def detect_structure_region(
    image_bgr: np.ndarray,
    cfg: Dict,
    detector: PlotRegionDetector,
    min_conf: float = 0.30,
) -> ChartStructure:
    """Run structure detection constrained to the detected plot region.

    Raises ``StructureDetectionError`` when the detector is unavailable or
    below ``min_conf`` so the caller can fall back to the classical-CV path.
    """
    region = detector.detect(image_bgr)
    if region is None or region.confidence < min_conf:
        raise StructureDetectionError("plot-region detector low confidence")
    crop, ox, oy, mask = apply_region(image_bgr, region)
    if crop.size == 0:
        raise StructureDetectionError("empty plot-region crop")
    struct = detect_structure(crop, cfg, exclude_mask=mask.astype(np.uint8))
    out = _offset_structure(struct, ox, oy)
    out.meta = {
        **(out.meta or {}),
        "region_backend": "plot_region",
        "region_conf": round(float(region.confidence), 3),
        "region_exclusions": [n for n, _b, _c in region.exclusions],
    }
    return out
