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

from ..schema import (
    AxisFitError,
    ExtractionResult,
    ExtractionError,
    StructureDetectionError,
)
from ..utils import read_image
from .base import TextBox
from .chart_structure import detect_structure
from .detector import YoloStructureDetector
from .coordinate_mapper import build_axes
from .curve_extractor import CurveExtractionError, extract_curves, extract_curves_multi
from .legend_matcher import match_legends
from .plot_region import (
    PlotRegionDetector,
    YoloPlotRegionDetector,
    detect_structure_region,
)
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
    "ocr_tier": "server",  # server (PP-OCRv5) | mobile (PP-OCRv4 fallback)
    "segmenter": "cv",  # cv | unet | multi_unet | auto
    "unet_checkpoint": "models/checkpoints/unet_curve.pt",
    "multi_unet_checkpoint": "models/checkpoints/unet_multi_curve.pt",
    "unet_size": 256,  # inference resolution (must match training resolution)
    "structure_backend": "cv",  # cv | yolo (Phase B-4)
    "yolo_weights": "models/detection/yolo_struct.pt",
    # Level-0 plot-region front-end (see 参考/方案_曲线提取结构检测组件化检测.md).
    # "none" keeps the behaviour identical to before this change; set to
    # "yolo" (and provide region_weights) to crop to the detected plot area
    # and mask out embedded tables / annotation text / captions before the
    # classical-CV structure detector runs.
    "region_backend": "none",  # none | yolo
    "region_weights": "models/detection/yolo_region.pt",
    "region_min_conf": 0.30,
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


def _build_ocr(backend: str, image_path: str, tier: str = "server") -> object:
    if backend == "stub":
        labels = os.path.splitext(image_path)[0] + "_labels.json"
        return StubOCRBackend(labels)
    if backend == "paddle":
        return PaddleOCRBackend(lang="en", device="auto", tier=tier)
    # auto: paddle if installed, else stub sidecar
    try:
        import paddleocr  # noqa: F401

        return PaddleOCRBackend(lang="en", device="auto", tier=tier)
    except ImportError:
        labels = os.path.splitext(image_path)[0] + "_labels.json"
        return StubOCRBackend(labels)


def _read_image_for_gate(image_path: str) -> Optional[np.ndarray]:
    """Best-effort image read for the quality gate on the failure path."""
    try:
        return read_image(image_path)
    except Exception:
        return None


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
        self._single_segmenter = None  # auto mode: single-curve U-Net
        self._multi_segmenter = None   # auto mode: multi-curve U-Net
        self._structure_detector = None
        self._region_detector: Optional[PlotRegionDetector] = None

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

    def _get_single_segmenter(self):
        """Single-curve U-Net (auto mode, cached per extractor instance)."""
        if self._single_segmenter is None:
            from .segmenter import UNetSegmenter

            self._single_segmenter = UNetSegmenter(
                self.cfg.get("unet_checkpoint"),
                size=int(self.cfg.get("unet_size", 512)),
            )
        return self._single_segmenter

    def _get_multi_segmenter(self):
        """Multi-curve U-Net (auto mode, cached per extractor instance)."""
        if self._multi_segmenter is None:
            from .segmenter import MultiUNetSegmenter

            self._multi_segmenter = MultiUNetSegmenter(
                self.cfg.get("multi_unet_checkpoint",
                             "models/checkpoints/unet_multi_curve.pt"),
                size=int(self.cfg.get("unet_size", 512)),
            )
        return self._multi_segmenter

    def _extract_auto(self, image, structure, x_axis, y_axis):
        """Auto single/multi-curve backend selection.

        Strategy (zero extra training): the multi-channel U-Net doubles as a
        curve COUNTER — the number of surviving channels IS the instance
        count (same instance-segmentation counting argument as LineFormer/
        LineEX).  >= 2 channels  -> multi result kept as-is;
        exactly 1 channel        -> re-extract with the (higher-precision)
        single-curve U-Net and keep whichever yields a curve;
        0 channels / model error -> single U-Net, then classical CV fallback.

        Returns ``(curves, backend_name)`` with backend in
        {"multi", "single", "cv"}.
        """
        def _multi():
            return extract_curves_multi(image, structure, x_axis, y_axis,
                                        self.cfg, self._get_multi_segmenter())

        def _single():
            return extract_curves(image, structure, x_axis, y_axis,
                                  self.cfg, segmenter=self._get_single_segmenter())

        def _cv():
            return extract_curves(image, structure, x_axis, y_axis, self.cfg)

        try:
            multi_curves = _multi()
        except Exception:
            multi_curves = []
        n_multi = len(multi_curves)
        if n_multi >= 2:
            return multi_curves, "multi"

        # 0 or 1 surviving channels: try the single-curve model.
        try:
            single_curves = _single()
        except Exception:
            single_curves = []
        if n_multi == 1 and len(single_curves) >= 1:
            return single_curves, "single"
        if n_multi == 0 and len(single_curves) >= 1:
            return single_curves, "single"
        if n_multi == 1:
            return multi_curves, "multi"  # only the multi path found anything

        # Nothing learned worked -> classical CV (training-free).
        try:
            return _cv(), "cv"
        except CurveExtractionError:
            raise

    # ------------------------------------------------------------------
    def _get_region_detector(self) -> Optional[PlotRegionDetector]:
        """Lazily build the Level-0 plot-region detector (or None).

        Returns None when ``region_backend`` is not ``yolo`` or when the
        weights are missing / ultralytics is unavailable, so the rest of the
        pipeline runs exactly as before.
        """
        if self._region_detector is not None:
            return self._region_detector
        self._region_detector = None
        if self.cfg.get("region_backend") == "yolo":
            try:
                det = YoloPlotRegionDetector(self.cfg.get("region_weights", ""))
                if det.available:
                    self._region_detector = det
            except Exception:
                self._region_detector = None
        return self._region_detector

    def _detect_structure_inner(self, image: np.ndarray):
        """Original structure branch (cv | yolo), kept for fallback."""
        if self.cfg.get("structure_backend") == "yolo":
            if self._structure_detector is None:
                from .detector import YoloStructureDetector

                self._structure_detector = YoloStructureDetector(
                    self.cfg.get("yolo_weights", ""),
                )
            return self._structure_detector.detect(image)
        return detect_structure(image, self.cfg)

    # ------------------------------------------------------------------
    def extract(self, image_path: str) -> ExtractionResult:
        """Run the pipeline; on failure re-raise the original ``ExtractionError``.

        When ``config.quality_gate`` is enabled (real-world robustness mode,
        see REAL_ROBUSTNESS_DESIGN.md), failures are additionally annotated
        with structured ``reject_code`` / ``reject_detail`` / ``quality``
        attributes so callers can triage A/B/C without parsing messages.
        """
        try:
            return self._extract_inner(image_path)
        except ExtractionError as e:
            # a stage may already have attached an explainable code (e.g.
            # categorical axis in _extract_inner); don't overwrite it
            if not getattr(e, "reject_code", None) and self.cfg.get("quality_gate"):
                from .quality_gate import classify_failure

                verdict = classify_failure(_read_image_for_gate(image_path), error=e)
                e.reject_code = verdict.reject_code  # type: ignore[attr-defined]
                e.reject_detail = verdict.reject_detail  # type: ignore[attr-defined]
                e.quality = verdict.quality  # type: ignore[attr-defined]
            raise

    def _extract_inner(self, image_path: str) -> ExtractionResult:
        timings: Dict[str, float] = {}
        warnings: List[str] = []
        t0 = time.time()

        image = read_image(image_path)
        timings["read_image"] = time.time() - t0

        # 1. structure
        t = time.time()
        region_detector = self._get_region_detector()
        if region_detector is not None:
            # Level-0 plot-region front-end: crop to the detected data area
            # and mask out embedded tables / annotation text, then run the
            # existing structure detector inside that clean region.  Any
            # detector failure falls back to the unchanged CV/YOLO path.
            try:
                structure = detect_structure_region(
                    image, self.cfg, region_detector,
                    min_conf=float(self.cfg.get("region_min_conf", 0.30)),
                )
            except StructureDetectionError:
                structure = self._detect_structure_inner(image)
        else:
            structure = self._detect_structure_inner(image)
        timings["structure"] = time.time() - t

        # 2. ticks
        t = time.time()
        ocr = _build_ocr(self.cfg["ocr_backend"], image_path,
                         tier=str(self.cfg.get("ocr_tier", "server")))
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
        try:
            x_axis, y_axis = build_axes(
                x_ticks, y_ticks,
                x_hint,
                y_hint,
                x_endpoints=(float(structure.y_axis_pixel), float(x1)),
                y_endpoints=(float(y0), float(structure.x_axis_pixel)),
            )
        except AxisFitError:
            # T2b: non-numeric (categorical) axis -> explainable reason
            # instead of a generic "not enough ticks" failure.
            from .quality_gate import CODE_OCR_CATEGORICAL_AXIS, is_categorical_axis
            from .quality_gate import _HINTS  # noqa: PLC2701 (stable hint table)

            if is_categorical_axis(x_ticks) or is_categorical_axis(y_ticks):
                exc = AxisFitError("categorical axis detected")
                exc.reject_code = CODE_OCR_CATEGORICAL_AXIS  # type: ignore[attr-defined]
                exc.reject_detail = _HINTS[CODE_OCR_CATEGORICAL_AXIS]  # type: ignore[attr-defined]
                exc.quality = "B"  # type: ignore[attr-defined]
                raise exc
            raise
        timings["axes"] = time.time() - t
        meta_titles = {k: {kk: vv for kk, vv in v.items() if kk != "center"}
                       for k, v in titles.items()}
        if x_axis.quality < 0.99:
            warnings.append(f"x-axis fit quality R^2={x_axis.quality:.4f}")
        if y_axis.quality < 0.99:
            warnings.append(f"y-axis fit quality R^2={y_axis.quality:.4f}")

        # 4. curve extraction (cv heuristics or learned U-Net segmentation)
        t = time.time()
        auto_backend_used = None
        seg_name = self.cfg.get("segmenter")
        if seg_name == "auto":
            curves, auto_backend_used = self._extract_auto(
                image, structure, x_axis, y_axis)
        elif seg_name == "multi_unet":
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
        if auto_backend_used is not None:
            meta["auto_segmenter"] = auto_backend_used

        # Real-world robustness: attach detected panels (only when enabled;
        # success path behaviour is otherwise unchanged).
        if self.cfg.get("quality_gate"):
            from .panel_detect import detect_panels

            try:
                structure.panels = detect_panels(image)
            except Exception:
                pass

        result = ExtractionResult(
            image_path=image_path,
            x_axis=x_axis,
            y_axis=y_axis,
            curves=curves,
            structure=structure,
            meta=meta,
            warnings=warnings,
        )

        # Real-world robustness: demote to B when an axis fit is weak
        # (linear/log judgement or mapping quality) — only when enabled.
        if self.cfg.get("quality_gate"):
            from .quality_gate import CODE_AXIS_TYPE_AMBIGUOUS, classify_success

            min_axis_q = float(self.cfg.get("quality_min_axis_r2", 0.95))
            worst_q = min(x_axis.quality, y_axis.quality)
            verdict = classify_success(axis_quality=worst_q if worst_q < min_axis_q else None)
            result.quality = verdict.quality
            result.status = verdict.status
            result.reject_code = verdict.reject_code
            result.reject_detail = verdict.reject_detail

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
