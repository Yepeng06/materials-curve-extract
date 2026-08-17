"""Tick reading: OCR of axis labels + association with tick marks.

Two OCR backends implement the same ``OCRBackend`` protocol:

* ``PaddleOCRBackend`` — production backend (PaddleOCR 3.x, lang='en'),
  GPU first with automatic CPU fallback.
* ``StubOCRBackend``   — reads ground-truth text boxes from a JSON sidecar
  (``<image_stem>_labels.json``) produced by ``scripts/gen_synthetic.py``.
  Used for deterministic tests and for ablating "perfect OCR" vs real OCR.

Association rule (single-plot, single-curve assumption):
  * a text box whose center lies below the bottom axis line is an x label;
  * a text box whose center lies left of the left axis line is a y label;
  * each label is matched to the nearest tick mark along the axis direction;
  * labels without a matching tick mark become ticks located at the label
    center (fallback when tick marks are not visible).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..schema import AxisRole, ChartStructure, Tick, TickReadingError
from ..utils import parse_number_text
from .base import TextBox


# ---------------------------------------------------------------------------
# OCR backends
# ---------------------------------------------------------------------------
class PaddleOCRBackend:
    """PaddleOCR 3.x text detection + recognition (tick-label oriented).

    ``strip_crops = True`` tells the tick reader to OCR only the two axis
    label strips (bottom + left) instead of the whole chart — the labels are
    small, so this is 10-50x faster than full-image OCR.
    """

    strip_crops = True
    _init_lock = threading.Lock()
    _shared: Dict[Tuple[str, str], "object"] = {}  # (lang, device) -> engine

    def __init__(self, lang: str = "en", device: str = "auto"):
        self.lang = lang
        self.device = device
        self._ocr = None

    def _ensure(self):
        if self._ocr is None:
            key = (self.lang, self.device)
            with PaddleOCRBackend._init_lock:  # 防多线程并发双初始化（Web 场景）
                cached = PaddleOCRBackend._shared.get(key)
                if cached is not None:
                    self._ocr = cached
                    return
                if self._ocr is not None:
                    return
                try:
                    # Load torch BEFORE paddle: paddleocr -> paddlex -> modelscope
                    # imports torch deep inside its chain, and on Windows loading
                    # torch's DLLs after paddle's DLLs are already in the process
                    # fails with WinError 127 on shm.dll.
                    import torch  # noqa: F401
                    import paddle

                    from paddleocr import PaddleOCR

                    kwargs = dict(
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        lang=self.lang,
                        enable_mkldnn=False,  # avoids oneDNN PIR conversion crashes
                        text_detection_model_name="PP-OCRv4_mobile_det",
                        text_recognition_model_name="PP-OCRv4_mobile_rec",
                    )
                    use_gpu = (
                        self.device == "gpu"
                        or (self.device == "auto" and paddle.device.is_compiled_with_cuda())
                    )
                    self._ocr = PaddleOCR(device="gpu" if use_gpu else "cpu", **kwargs)
                    self._device_used = "gpu" if use_gpu else "cpu"
                    PaddleOCRBackend._shared[key] = self._ocr
                except ImportError as e:  # pragma: no cover
                    raise TickReadingError(
                        "PaddleOCR is not installed; use --ocr stub or install "
                        "paddlepaddle + paddleocr"
                    ) from e

    def read_text_boxes(self, image_bgr: np.ndarray) -> List[TextBox]:
        self._ensure()
        res = self._ocr.predict(image_bgr)
        boxes: List[TextBox] = []
        for r in res:
            texts = r.get("rec_texts") or []
            scores = r.get("rec_scores") or []
            polys = r.get("rec_polys") or r.get("dt_polys") or []
            for i, t in enumerate(texts):
                if i < len(polys):
                    box = np.asarray(polys[i], dtype=np.float64)
                else:
                    continue
                score = float(scores[i]) if i < len(scores) else 1.0
                if t and t.strip():
                    boxes.append(TextBox(box=box, text=t.strip(), score=score))
        return boxes


class StubOCRBackend:
    """Ground-truth text boxes from a JSON sidecar (see gen_synthetic.py)."""

    def __init__(self, labels_path: str):
        self.labels_path = labels_path

    def read_text_boxes(self, image_bgr: np.ndarray) -> List[TextBox]:
        if not os.path.exists(self.labels_path):
            raise TickReadingError(f"stub OCR labels file not found: {self.labels_path}")
        with open(self.labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        boxes: List[TextBox] = []
        for item in data:
            box = np.asarray(item["box"], dtype=np.float64)
            boxes.append(TextBox(box=box, text=str(item["text"]),
                                 score=float(item.get("score", 1.0)), anchored=True))
        return boxes


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------
def _associate(
    tick_px: List[float],
    labels: List[TextBox],
    axis: str,  # "x" | "y"
    tol: float,
    drop_unmatched: bool = False,
) -> List[Tick]:
    """Globally assign labels to tick marks by nearest distance.

    All (tick, label) pairs within ``tol`` are considered and assigned in
    ascending distance order, so a tick cannot steal a label that matches a
    later tick much better (a failure mode of naive per-tick greedy search
    when minor ticks are dense).

    ``drop_unmatched`` (B-5a): when the axis has plenty of detected tick
    marks (>= 4), a label that matches NO mark is almost certainly stray
    axis-title/legend text ('10' fragments of a y-axis title like
    'Creep strain (10...' -- img_0013 regression) and must NOT become a
    fallback tick, whose stray pixel poisons both the value sequence and
    the pixel-spacing vote.
    """
    axis_i = 0 if axis == "x" else 1
    pairs = []
    for i, p in enumerate(tick_px):
        for j, lb in enumerate(labels):
            d = abs(lb.center[axis_i] - p)
            if d < tol:
                pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])

    tick_label: dict = {}  # tick idx -> label idx
    used_labels: set = set()
    for _, i, j in pairs:
        if i in tick_label or j in used_labels:
            continue
        tick_label[i] = j
        used_labels.add(j)

    ticks: List[Tick] = []
    for i, p in enumerate(tick_px):
        j = tick_label.get(i)
        if j is not None:
            lb = labels[j]
            pixel = lb.center[axis_i] if lb.anchored else p
            ticks.append(Tick(pixel=pixel, value=parse_number_text(lb.text),
                              text=lb.text, score=lb.score))
        else:
            ticks.append(Tick(pixel=p, value=None, text="", score=0.0))
    # drop_unmatched only when the marks are trustworthy (>= 2 labels
    # actually matched): if nearly nothing matched, the tick marks were
    # misdetected and the label centres must carry the axis (fail_001 /
    # img_0008 scenario -- strict mode would throw every real label away).
    if not drop_unmatched or len(used_labels) < 2:
        for j, lb in enumerate(labels):
            if j not in used_labels:
                ticks.append(
                    Tick(
                        pixel=float(lb.center[axis_i]),
                        value=parse_number_text(lb.text),
                        text=lb.text,
                        score=lb.score,
                    )
                )
    ticks.sort(key=lambda t: t.pixel)
    return ticks


def _dedupe_ticks(ticks: List[Tick], tol: float = 10.0) -> List[Tick]:
    """Collapse near-coincident ticks (< ``tol`` px) to one.

    B-5a: PP-OCRv4 can emit two boxes for ONE label (e.g. '10-1' and '0-1'
    5 px apart at the same 10^-2 tick -- img_0051), which then pollutes the
    value-sequence disambiguation and the RANSAC pre-fit with a duplicate.
    Keeps the tick that has a value (else the higher score).
    """
    out: List[Tick] = []
    for t in sorted(ticks, key=lambda t: t.pixel):
        if out and abs(t.pixel - out[-1].pixel) < tol:
            o = out[-1]
            if (t.value is not None and o.value is None) or (
                t.value is not None and o.value is not None and t.score > o.score
            ) or (t.value is None and o.value is None and t.score > o.score):
                out[-1] = t
        else:
            out.append(t)
    return out


def _strip_crop(image_bgr: np.ndarray, structure: ChartStructure,
                axis: str) -> Tuple[np.ndarray, int, int]:
    """Crop the x- or y-axis label strip; returns (crop, offset_x, offset_y).

    y-strip width must NOT depend on y_axis_col (a misdetected axis column
    collapses it to a few px — the fail_001 root cause); use a generous
    left band of the image instead.
    """
    h, w = image_bgr.shape[:2]
    x0, y0, _, _ = structure.plot_bbox
    x_axis_row = structure.x_axis_pixel
    y_axis_col = structure.y_axis_pixel
    if axis == "x":
        strip_h = min(h - x_axis_row - 1, max(80, h // 8))
        crop = image_bgr[x_axis_row - 6 : x_axis_row + strip_h,
                         max(0, y_axis_col - 10) :, :]
        return crop, max(0, y_axis_col - 10), x_axis_row - 6
    strip_w = min(w, max(40, int(w * 0.30)))
    crop = image_bgr[max(0, y0 - 20) : min(h, x_axis_row + 20), :strip_w, :]
    return crop, 0, max(0, y0 - 20)


def _ocr_strip_scaled(image_bgr: np.ndarray, structure: ChartStructure,
                      ocr: "object", axis: str, scale: int) -> List[TextBox]:
    """OCR one axis strip at a scale; boxes in full-image coordinates."""
    crop, ox, oy = _strip_crop(image_bgr, structure, axis)
    if crop.size == 0:
        return []
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    boxes: List[TextBox] = []
    for b in ocr.read_text_boxes(up):
        box = b.box / scale
        box[:, 0] += ox
        box[:, 1] += oy
        boxes.append(TextBox(box=box, text=b.text, score=b.score))
    return boxes


def _ocr_strips(image_bgr: np.ndarray, structure: ChartStructure, ocr: "object",
                cfg: Dict) -> List[TextBox]:
    """OCR the x- and y-axis label strips only (fast path for PaddleOCR).

    Returns text boxes in full-image coordinates.  Falls back to a single
    full-image OCR call when the backend does not declare strip support.
    """
    if not getattr(ocr, "strip_crops", False):
        return ocr.read_text_boxes(image_bgr)
    boxes: List[TextBox] = []
    for axis in ("x", "y"):
        boxes.extend(_ocr_strip_scaled(image_bgr, structure, ocr, axis, 2))
    return boxes


def _merge_boxes(a: List[TextBox], b: List[TextBox], tol: float = 10.0) -> List[TextBox]:
    """Merge two box lists, keeping a's boxes on near-duplicates."""
    out = list(a)
    for bx in b:
        cx, cy = bx.center
        if all((cx - o.center[0]) ** 2 + (cy - o.center[1]) ** 2 > tol * tol
               for o in out):
            out.append(bx)
    return out


def _classify_labels(
    boxes: List[TextBox], structure: ChartStructure
) -> Tuple[List[TextBox], List[TextBox]]:
    """Split OCR boxes into x- and y-tick labels by region rules.

    Region rules (loose on purpose -- non-numeric text simply fails to parse
    later and is harmless):

    * x label: center below the bottom axis line, within the plot width
      (plus a 15% margin on each side).
    * y label: center inside the left 25% of the plot width, above the
      bottom axis line, and below the title zone (title sits further above
      the plot top).  Deliberately does NOT depend on the absolute
      y_axis_col boundary -- a misdetected axis column must not reject
      every y label (fail_001 root cause).
    """
    x0, y0, x1, y1 = structure.plot_bbox
    x_axis_row = structure.x_axis_pixel
    plot_w = max(1, x1 - x0)
    plot_h = max(1, x_axis_row - y0)

    x_labels = [
        b for b in boxes
        if b.center[1] > x_axis_row + 2
        and x0 - 0.15 * plot_w <= b.center[0] <= x1 + 0.15 * plot_w
    ]
    y_labels = [
        b for b in boxes
        if b.center[0] < x0 + 0.25 * plot_w
        and b.center[1] < x_axis_row + 10  # allow labels hugging the axis row
        and b.center[1] > y0 - 0.3 * plot_h
    ]
    return x_labels, y_labels


def _is_tick_label(b: TextBox, min_score: float) -> bool:
    """Keep an OCR box as a tick-label candidate.

    B-5a filters:
      * rec score below ``min_score`` -- single-glyph fragments ('1'@0.1,
        '0'@0.5) that pollute the value sequence;
      * long non-numeric text -- axis titles ('Creep strain (%)' fragments
        like '( s da') and legend strings are never tick labels, and their
        stray positions poison the pixel-spacing vote (img_0061 regression).
    """
    if b.score < min_score:
        return False
    t = b.text.strip()
    if not t:
        return False
    digits = sum(c.isdigit() for c in t)
    letters = sum(c.isalpha() for c in t)
    if letters >= 2 and digits == 0:
        return False
    return True

def read_ticks(
    image_bgr: np.ndarray,
    structure: ChartStructure,
    ocr: "object",
    cfg: Optional[Dict] = None,
) -> Tuple[List[Tick], List[Tick]]:
    """OCR the chart (or its axis strips), classify labels by axis, associate.

    Whole-image OCR fallback: when a strip-based backend yields fewer than
    two valued ticks on either axis (strip crops missed the labels), re-run
    OCR on the full image, merge the boxes and re-classify.  The fail_001
    sample reads every tick label correctly on the full image.
    """
    cfg = cfg or {}
    tol = float(cfg.get("tick_assoc_tol_px", 80))
    min_score = float(cfg.get("tick_min_score", 0.55))
    can_strip = bool(getattr(ocr, "strip_crops", False))
    # B-5a: drop low-confidence OCR boxes before association -- PP-OCRv4
    # emits single-glyph fragments ('1'@0.1, '0'@0.5, '6'@0.4) that would
    # otherwise pollute the value sequence with fake ticks.  Genuine tick
    # labels score >= 0.85 on the platform set (lowest observed 0.71).
    boxes = [b for b in _ocr_strips(image_bgr, structure, ocr, cfg)
             if _is_tick_label(b, min_score)]

    x_ticks: List[Tick] = []
    y_ticks: List[Tick] = []
    # B-5a: when an axis has >= 4 detected tick marks, labels that match
    # no mark are stray title/legend text and are dropped (no fallback tick)
    strict_x = len(structure.x_ticks_px) >= 4
    strict_y = len(structure.y_ticks_px) >= 4
    for attempt in (0, 1, 2):
        x_labels, y_labels = _classify_labels(boxes, structure)
        x_ticks = _associate(structure.x_ticks_px, x_labels, "x", tol, strict_x)
        y_ticks = _associate(structure.y_ticks_px, y_labels, "y", tol, strict_y)
        nv_x = sum(1 for t in x_ticks if t.value is not None)
        nv_y = sum(1 for t in y_ticks if t.value is not None)

        def _enough(nv, marks):
            # >= 2 valued ticks, or (few read but many tick marks detected ->
            # the strip OCR likely missed labels; a whole-image pass may help)
            return nv >= 2 and not (nv <= 2 and len(marks) >= 4)

        if not can_strip or (
            _enough(nv_x, structure.x_ticks_px)
            and _enough(nv_y, structure.y_ticks_px)
        ):
            break
        if attempt == 0:
            full = [b for b in ocr.read_text_boxes(image_bgr)
                    if _is_tick_label(b, min_score)]
            boxes = _merge_boxes(boxes, full)
        elif attempt == 1:
            # B-5a: high-resolution (4x) strip re-OCR -- the 10^N
            # superscript family ('10', '100', '10-', '012'...) mostly
            # resolves at 4x, and labels missed at 2x get detected.  Only
            # run for axes that are still short on valued ticks.
            for axis, tks, marks in (
                ("x", x_ticks, structure.x_ticks_px),
                ("y", y_ticks, structure.y_ticks_px),
            ):
                nv = sum(1 for t in tks if t.value is not None)
                if not _enough(nv, marks):
                    hi = [b for b in _ocr_strip_scaled(
                        image_bgr, structure, ocr, axis, 4)
                        if _is_tick_label(b, min_score)]
                    boxes = _merge_boxes(boxes, hi)
    if not boxes:
        raise TickReadingError("OCR returned no text boxes")
    return _dedupe_ticks(x_ticks), _dedupe_ticks(y_ticks)
