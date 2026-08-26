# -*- coding: utf-8 -*-
"""Extract curves from user-picked real images."""
import sys, os, glob, json
sys.path.insert(0, r"F:\CODE\New\baseline\src")
import numpy as np
from mci.pipeline.extractor import Extractor
from mci.utils import read_image

files = sorted(glob.glob(r"F:\CODE\New\temp\*.png"))
ext = Extractor(ocr_backend="paddle", segmenter="multi_unet",
                config_override={"quality_gate": True})
out = []
for fp in files:
    name = os.path.basename(fp)
    print(f"\n===== {name} =====", flush=True)
    try:
        res = ext.extract(fp)
        meta_t = res.meta.get("titles", {})
        print(f"quality={res.quality} status={res.status} code={res.reject_code}")
        print(f"curves={len(res.curves)} x={res.x_axis.kind.value} y={res.y_axis.kind.value} "
              f"axis_q=({res.x_axis.quality:.4f},{res.y_axis.quality:.4f})")
        if res.reject_detail: print("hint:", res.reject_detail)
        t = meta_t.get("title") or {}
        xl = meta_t.get("x_label") or {}
        yl = meta_t.get("y_label") or {}
        print(f"title={t.get('text','')!r} x_label={xl.get('text','')!r} y_label={yl.get('text','')!r}")
        for i, c in enumerate(res.curves[:6]):
            print(f"  curve{i+1}: {len(c.points)} pts label={c.legend_label!r}")
        # save overlay
        import cv2
        img = read_image(fp)
        for c in res.curves:
            for (px, py) in c.pixel_points[::2]:
                cv2.circle(img, (int(px), int(py)), 2, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(r"F:\CODE\New\temp", "overlay_" + name), img)
        out.append({"file": name, "quality": res.quality, "status": res.status,
                    "code": res.reject_code, "n_curves": len(res.curves),
                    "x": res.x_axis.kind.value, "y": res.y_axis.kind.value})
    except Exception as e:
        q = getattr(e, "quality", "?")
        code = getattr(e, "reject_code", None)
        detail = getattr(e, "reject_detail", None)
        print(f"FAILED quality={q} code={code}")
        if detail: print("hint:", detail)
        print(f"error: {type(e).__name__}: {e}")
        out.append({"file": name, "quality": q, "status": "failed",
                    "code": code, "error": str(e)[:150]})

print("\n===== SUMMARY =====")
for r in out:
    print(json.dumps(r, ensure_ascii=False))
