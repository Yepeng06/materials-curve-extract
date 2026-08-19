"""End-to-end extraction orchestrator.

Loads config, wires the modules, runs the pipeline and collects diagnostics
(stage timings, warnings, debug images).  ``Extractor.extract`` is the single
public entry point used by scripts, tests and the future web backend.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from ..schema import ExtractionResult, ExtractionError
from ..utils import read_image
from .base import TextBox
from .chart_structure import detect_structure
from .detector import YoloStructureDetector
from .coordinate_mapper import build_axes
from .curve_extractor import extract_curves
from .legend_matcher import match_legends
from .tick_reader import PaddleOCRBackend, StubOCRBackend, read_ticks
from .title_reader import read_rotated_y_title, read_titles

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "configs", "baseline.yaml",
)

DEFAULTS = {
    "axis_frac_threshold": 0.45,
    "tick_assoc_tol_px": 25,  # label<->mark pairing tolerance
    "tick_min_score": 0.55,  # B-5a: drop OCR boxes below this rec score
    "min_curve_area": 60,
    "max_points": 2000,
    "x_kind_hint": "auto",
    "y_kind_hint": "auto",
    "ocr_backend": "auto",  # auto | paddle | stub
    "segmenter": "cv",  # cv | unet | multi_unet
    "unet_checkpoint": "models/checkpoints/unet_curve.pt",
    "multi_unet_checkpoint": "models/checkpoints/unet_multi_curve.pt",
    "unet_size": 256,  # inference resolution (must match training resolution)
    "structure_backend": "cv",  # cv | yolo (Phase B-4)
    "yolo_weights": "models/detection/yolo_struct.pt",
    "debug": False,
}


def load_config(path: Optional[str] = None) -> Dict:
    cfg = dict(DEFAULTS)
    p = path or DEFAULT_CONFIG_PATH
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    return cfg


def _build_ocr(backend: str, image_path: str) -> object:
    if backend == "stub":
        labels = os.path.splitext(image_path)[0] + "_labels.json"
        return StubOCRBackend(labels)
    if backend == "paddle":
        return PaddleOCRBackend(lang="en", device="auto")
    # auto: paddle if installed, else stub sidecar
    try:
        import paddleocr  # noqa: F401

        return PaddleOCRBackend(lang="en", device="auto")
    except ImportError:
        labels = os.path.splitext(image_path)[0] + "_labels.json"
        return StubOCRBackend(labels)


class Extractor:
    def __init__(self, config_path: Optional[str] = None, ocr_backend: Optional[str] = None,
                 segmenter: Optional[str] = None, debug_dir: Optional[str] = None,
                 config_override: Optional[Dict] = None):
        self.cfg = load_config(config_path)
        if config_override:
            self.cfg.update(config_override)
        if ocr_backend:
            self.cfg["ocr_backend"] = ocr_backend
        if segmenter:
            self.cfg["segmenter"] = segmenter
        self.debug_dir = debug_dir
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
        self._segmenter = None
        self._structure_detector = None

    def _get_segmenter(self):
        if self._segmenter is None and self.cfg.get("segmenter") == "unet":
            from .segmenter import UNetSegmenter

            self._segmenter = UNetSegmenter(
                self.cfg.get("unet_checkpoint"),
                size=int(self.cfg.get("unet_size", 256)),
            )
        elif self._segmenter is None and self.cfg.get("segmenter") == "multi_unet":
            from .segmenter import MultiUNetSegmenter

            self._segmenter = MultiUNetSegmenter(
                self.cfg.get("multi_unet_checkpoint", "models/checkpoints/unet_multi_curve.pt"),
                size=int(self.cfg.get("unet_size", 256)),
            )
        return self._segmenter

    # ------------------------------------------------------------------
    def extract(self, image_path: str) -> ExtractionResult:
        timings: Dict[str, float] = {}
        warnings: List[str] = []
        t0 = time.time()

        image = read_image(image_path)
        timings["read_image"] = time.time() - t0

        # 1. structure
        t = time.time()
        if self.cfg.get("structure_backend") == "yolo":
            if self._structure_detector is None:
                from .detector import YoloStructureDetector

                self._structure_detector = YoloStructureDetector(
                    self.cfg.get("yolo_weights", ""),
                )
            structure = self._structure_detector.detect(image)
        else:
            structure = detect_structure(image, self.cfg)
        timings["structure"] = time.time() - t

        # 2. ticks
        t = time.time()
        ocr = _build_ocr(self.cfg["ocr_backend"], image_path)
        try:
            x_ticks, y_ticks = read_ticks(image, structure, ocr, self.cfg)
        except ExtractionError as e:
            raise
        timings["ticks"] = time.time() - t
        n_val_x = sum(1 for tk in x_ticks if tk.value is not None)
        n_val_y = sum(1 for tk in y_ticks if tk.value is not None)
        if n_val_x < 2:
            warnings.append(f"x-axis: only {n_val_x} readable tick labels")
        if n_val_y < 2:
            warnings.append(f"y-axis: only {n_val_y} readable tick labels")

        # 3. coordinate mapping (linear/log auto detection)
        t = time.time()
        # endpoint pixels (x: axis line left/right; y: plot top/bottom) feed
        # the 0-start endpoint anchor in fit_axis
        x0, y0, x1, y1 = structure.plot_bbox
        # Phase B-2: whole-image text gives title / axis labels / units and
        # a log prior for the kind judgement (only when OCR provides boxes;
        # stub sidecars carry tick labels only, so titles stay empty there)
        # whole-image OCR boxes are shared by the title reader and the
        # legend matcher (one OCR call instead of two)
        full_boxes: Optional[List[TextBox]] = None

        def get_full_boxes() -> List[TextBox]:
            nonlocal full_boxes
            if full_boxes is None:
                full_boxes = ocr.read_text_boxes(image)
            return full_boxes

        titles: dict = {}
        try:
            titles = read_titles(get_full_boxes(), structure)
            # vertical y-axis titles need a rotated OCR pass; use it when
            # the horizontal pass missed the label or parsed no unit
            yl = titles.get("y_label")
            if yl is None or not yl.get("unit") or not yl.get("variable"):
                rot = read_rotated_y_title(image, structure, ocr)
                if rot and rot.get("text"):
                    titles["y_label"] = rot
        except Exception:
            titles = {}
        x_hint = titles.get("x_label", {}).get("log_hint", "") or self.cfg.get("x_kind_hint", "auto")
        y_hint = titles.get("y_label", {}).get("log_hint", "") or self.cfg.get("y_kind_hint", "auto")
        x_axis, y_axis = build_axes(
            x_ticks, y_ticks,
            x_hint,
            y_hint,
            x_endpoints=(float(structure.y_axis_pixel), float(x1)),
            y_endpoints=(float(y0), float(structure.x_axis_pixel)),
        )
        timings["axes"] = time.time() - t
        meta_titles = {k: {kk: vv for kk, vv in v.items() if kk != "center"}
                       for k, v in titles.items()}
        if x_axis.quality < 0.99:
            warnings.append(f"x-axis fit quality R^2={x_axis.quality:.4f}")
        if y_axis.quality < 0.99:
            warnings.append(f"y-axis fit quality R^2={y_axis.quality:.4f}")

        # 4. curve extraction (cv heuristics or learned U-Net segmentation)
        t = time.time()
        if self.cfg.get("segmenter") == "multi_unet":
            from .curve_extractor import extract_curves_multi

            curves = extract_curves_multi(image, structure, x_axis, y_axis,
                                          self.cfg, self._get_segmenter())
        else:
            curves = extract_curves(image, structure, x_axis, y_axis, self.cfg,
                                    segmenter=self._get_segmenter())
        timings["curve"] = time.time() - t

        # 5. legend matching (baseline no-op)
        t = time.time()
        curves = match_legends(curves, structure, get_full_boxes(), self.cfg,
                                      image_bgr=image)
        timings["legend"] = time.time() - t

        timings["total"] = time.time() - t0
        meta = {"timings": timings, "ocr_backend": type(ocr).__name__,
                "titles": meta_titles}

        result = ExtractionResult(
            image_path=image_path,
            x_axis=x_axis,
            y_axis=y_axis,
            curves=curves,
            structure=structure,
            meta=meta,
            warnings=warnings,
        )

        if self.debug_dir:
            self._save_debug(image, result, ocr)
        return result

    # ------------------------------------------------------------------
    def _save_debug(self, image: np.ndarray, result: ExtractionResult, ocr: object) -> None:
        stem = os.path.splitext(os.path.basename(result.image_path))[0]
        dbg = image.copy()
        x0, y0, x1, y1 = result.structure.plot_bbox
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for px in result.structure.x_ticks_px:
            cv2.drawMarker(dbg, (int(px), result.structure.y_axis_pixel), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
        for py in result.structure.y_ticks_px:
            cv2.drawMarker(dbg, (result.structure.x_axis_pixel, int(py)), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
        for c in result.curves:
            for (px, py) in c.pixel_points[::3]:
                cv2.circle(dbg, (px, py), 2, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.debug_dir, f"{stem}_structure.png"), dbg)

        # OCR boxes
        dbg2 = image.copy()
        try:
            boxes = ocr.read_text_boxes(image)
        except Exception:
            boxes = []
        for b in boxes:
            pts = b.box.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(dbg2, [pts], True, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(self.debug_dir, f"{stem}_ocr.png"), dbg2)
