"""Module interfaces.

The baseline implements each interface with classical computer vision.
Later stages replace the implementations one-by-one:

- ``ChartStructureDetector``  -> YOLOv8-nano (ultralytics) detection model
- ``OCRBackend``              -> already PaddleOCR in the baseline
- ``CurveSegmenter``          -> U-Net (PyTorch) binary segmentation
- ``LegendMatcher``           -> PaddleOCR + color/space association (multi-curve)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, Tuple

import numpy as np

from ..schema import ChartStructure, Curve, Tick


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
@dataclass
class TextBox:
    """One OCR result: 4-point polygon (x, y), text and confidence.

    ``anchored``: True when the box position is ground-truth anchored (stub
    backend) — the tick association then uses the box center as the tick
    pixel.  For real OCR (anchored=False) the detected tick mark pixel is
    used, since OCR glyph centers are only approximate.
    """

    box: np.ndarray  # shape (4, 2), order: tl, tr, br, bl (PaddleOCR convention)
    text: str
    score: float
    anchored: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        return tuple(self.box.mean(axis=0))  # (cx, cy)


class OCRBackend(Protocol):
    """Reads all text boxes from an image (BGR uint8 array)."""

    def read_text_boxes(self, image_bgr: np.ndarray) -> List[TextBox]: ...


# ---------------------------------------------------------------------------
# Chart structure
# ---------------------------------------------------------------------------
class ChartStructureDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> ChartStructure: ...


# ---------------------------------------------------------------------------
# Tick reading
# ---------------------------------------------------------------------------
class TickReader(Protocol):
    """Associates OCR'd labels with tick marks; returns ticks for both axes."""

    def read(self, image_bgr: np.ndarray, structure: ChartStructure) -> Tuple[List[Tick], List[Tick]]: ...


# ---------------------------------------------------------------------------
# Curve segmentation (future: U-Net)
# ---------------------------------------------------------------------------
class CurveSegmenter(Protocol):
    """Maps a plot-region image to a binary curve mask.

    The baseline implements this with morphology + connected components;
    the U-Net version returns the same ``np.ndarray`` mask (0/1, uint8).
    """

    def segment(self, plot_region_bgr: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Legend matching (future: multi-curve)
# ---------------------------------------------------------------------------
class LegendMatcher(Protocol):
    """Assigns legend labels to curves. Baseline: identity (single curve)."""

    def match(
        self,
        curves: List[Curve],
        structure: ChartStructure,
        text_boxes: List[TextBox],
    ) -> List[Curve]: ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Flat config merged from configs/baseline.yaml (dict-compatible)."""

    values: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.values.get(key, default)
