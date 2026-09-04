"""Tick-label OCR accuracy evaluation (per-label crop, 4x upscale).

Compares ``parse_number_text(OCR(text))`` against the ground-truth numeric
value parsed from the synthetic GT sidecar ``*_labels.json``.  Reports
overall accuracy plus the superscript subset (10^N family), which is the
known weak spot (goal.md 任务1.4: 上标/下标/公式/手写识别改进).

Usage (conda env mci, from baseline/):
    python scripts/eval_tick_ocr.py --dir data/synthetic --tier server --limit 20
    python scripts/eval_tick_ocr.py --dir data/synthetic --tier mobile
    python scripts/eval_tick_ocr.py --dir web/examples --tier server
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mci.pipeline.tick_reader import PaddleOCRBackend  # noqa: E402
from mci.utils import parse_number_text  # noqa: E402


def _load_sidecars(folder: str):
    """Yield (image_path, [(text, box), ...]) from GT label sidecars."""
    for name in sorted(os.listdir(folder)):
        if not name.endswith("_labels.json"):
            continue
        stem = os.path.join(folder, name[: -len("_labels.json")])
        img_path = stem + ".png"
        if not os.path.isfile(img_path):
            continue
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            items = json.load(f)
        boxes = []
        for it in items:
            text = str(it.get("text", "")).strip()
            box = it.get("box")
            if text and box and len(box) >= 4:
                boxes.append((text, np.array(box, dtype=np.float32)))
        if boxes:
            yield img_path, boxes


def _gt_value(text: str):
    """GT numeric value; GT text like '10^-2' / unicode superscripts."""
    v = parse_number_text(text)
    return v


def _crop_pad(img, box, pad_frac=0.18, canvas_frac=0.55):
    """Crop the label with margins, then paste onto a white canvas.

    PP-OCR server det fails on tiny single-word crops (no document
    context — the word fills the whole frame); padding to a canvas where
    the word occupies <= canvas_frac of the width restores detection.
    """
    xs, ys = box[:, 0], box[:, 1]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pw = max(2, int((x1 - x0) * pad_frac))
    ph = max(2, int((y1 - y0) * pad_frac))
    H, W = img.shape[:2]
    crop = img[max(0, y0 - ph): min(H, y1 + ph),
               max(0, x0 - pw): min(W, x1 + pw)]
    ch, cw = crop.shape[:2]
    scale = min(1.0, canvas_frac / max(cw / 640.0, 0.001))  # no-op guard
    target_w = max(int(cw / canvas_frac), 96)
    target_h = max(int(ch / canvas_frac), 64)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    oy = (target_h - ch) // 2
    ox = (target_w - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="folder with *_labels.json + *.png")
    ap.add_argument("--tier", default="server", choices=["server", "mobile"])
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    ap.add_argument("--out", default="", help="write per-label report JSON")
    args = ap.parse_args()

    backend = PaddleOCRBackend(lang="en", device="auto", tier=args.tier)
    rows = []
    n_imgs = 0
    t0 = time.time()
    for img_path, boxes in _load_sidecars(args.dir):
        if args.limit and n_imgs >= args.limit:
            break
        n_imgs += 1
        img = cv2.imread(img_path)
        if img is None:
            continue
        for text, box in boxes:
            gt = _gt_value(text)
            if gt is None:
                continue  # non-numeric GT (titles etc.) — out of scope
            crop = _crop_pad(img, box)
            # 4x upscale: small tick glyphs are the weak spot of mobile rec
            crop = cv2.resize(crop, None, fx=4.0, fy=4.0,
                              interpolation=cv2.INTER_CUBIC)
            try:
                boxes_pred = backend.read_text_boxes(crop)
            except Exception as e:  # noqa: BLE001
                rows.append({"img": os.path.basename(img_path), "gt_text": text,
                             "pred": "", "gt": gt, "ok": False, "err": str(e)})
                continue
            pred = " ".join(b.text for b in boxes_pred).strip()
            got = parse_number_text(pred)
            ok = got is not None and abs(got - gt) <= 1e-6 * max(1.0, abs(gt))
            rows.append({"img": os.path.basename(img_path), "gt_text": text,
                         "pred": pred, "gt": gt, "got": got, "ok": ok})
    dt = time.time() - t0

    total = len(rows)
    n_ok = sum(1 for r in rows if r.get("ok"))
    is_sup = lambda t: ("10" in t and any(c.isdigit() for c in t)
                        and (t.strip().startswith("10") or "x10" in t or "×10" in t))
    sup_rows = [r for r in rows if is_sup(r["gt_text"])]
    sup_ok = sum(1 for r in sup_rows if r.get("ok"))
    print(f"\n=== tick OCR eval ({args.tier}) ===")
    print(f"images: {n_imgs}   labels: {total}   time: {dt:.1f}s")
    if total:
        print(f"overall accuracy : {n_ok}/{total} = {n_ok / total:.3f}")
    if sup_rows:
        print(f"superset 10^N    : {sup_ok}/{len(sup_rows)} = {sup_ok / len(sup_rows):.3f}")
    bad = [r for r in rows if not r.get("ok")][:20]
    for r in bad:
        print(f"  MISS {r['img']}  gt_text={r['gt_text']!r}  pred={r.get('pred','')!r}"
              f"  gt={r['gt']}  got={r.get('got')}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"tier": args.tier, "images": n_imgs, "rows": rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
