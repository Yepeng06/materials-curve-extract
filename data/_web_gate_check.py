import sys, os
sys.path.insert(0, r"F:\CODE\New\baseline\src")
from mci.pipeline.extractor import Extractor

# pick one val image with stub labels
import glob
cands = glob.glob(r"F:\CODE\New\baseline\data\val_multi\*.png")[:1]
if not cands:
    print("no val image"); sys.exit(0)
img = cands[0]
ext = Extractor(ocr_backend="stub", segmenter="cv", config_override={"quality_gate": True})
res = ext.extract(img)
print("quality:", res.quality, "status:", res.status, "reject_code:", res.reject_code)
print("curves:", len(res.curves), "panels:", len(res.structure.panels) if res.structure else "?")
print("x_axis:", res.x_axis.kind.value if res.x_axis else None, "y_axis:", res.y_axis.kind.value if res.y_axis else None)
print("SUCCESS-PATH-OK" if res.quality == "A" and len(res.curves) > 0 else "CHECK")
