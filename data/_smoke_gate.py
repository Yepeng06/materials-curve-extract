# -*- coding: utf-8 -*-
"""Smoke test: A/B/C triage on real images (R1e subset)."""
import sys, os, glob
sys.path.insert(0, r"F:\CODE\New\baseline\src")
import numpy as np
import cv2

from mci.pipeline.quality_gate import classify_failure, classify_success, is_categorical_axis
from mci.pipeline.panel_detect import detect_panels

ROOT = r"F:\CODE\New\baseline\data\real_diag\images"
subset = [l.strip() for l in open(os.path.join(ROOT, "eval_subset.txt"), encoding="utf-8") if l.strip()]
print("subset size:", len(subset))

# pick a few representative images by QC score range
import csv
manifest = {r["image"]: r for r in csv.DictReader(open(os.path.join(ROOT, "manifest.csv"), encoding="utf-8"))}

def triage(name):
    img = cv2.imread(os.path.join(ROOT, name))
    if img is None: return None
    panels = detect_panels(img)
    # simulate pipeline failure variants
    dark = (img.mean(axis=2) < 60).mean()
    return len(panels), dark, name

results = []
for name in subset[:12]:
    r = triage(name)
    if r: results.append(r)

for n_panels, dark_frac, name in results:
    # white img for classify (dark check needs BGR)
    print(f"{name:50s} panels={n_panels} dark={dark_frac:.2f}")
print("\n--- 326-library re-check (multi-panel rate) ---")
from collections import Counter
import glob as g
cnt = Counter()
for fp in g.glob(r"F:\CODE\New\dataset\curves_final\*.png"):
    img = cv2.imread(fp)
    cnt[len(detect_panels(img)) >= 2] += 1
print("multi:", cnt.get(True, 0), "single/zero:", cnt.get(False, 0))
