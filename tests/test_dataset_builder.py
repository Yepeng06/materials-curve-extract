"""Tests for data/dataset_builder.py (materials-curve-dataset-platform adapter).

Skipped automatically when the platform checkout is not present.  Generates a
tiny set and validates the baseline sidecar contract (PNG / mask / CSV /
meta.json / labels.json / MCG-JSON) and the eval-readiness of single-curve
samples (evaluate.py-compatible meta + csv + labels).
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pytest

_PLATFORM_ROOT = r"F:\CLAUDE\NewProject1\materials-curve-dataset-platform"
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_PLATFORM_ROOT),
    reason="materials-curve-dataset-platform not present",
)


def _run_builder(tmp_path, extra_args: list[str]) -> int:
    sys.path.insert(0, _DATA_DIR)
    import dataset_builder

    args = ["--out-dir", str(tmp_path), "--count", "3", "--seed", "777",
            "--platform-root", _PLATFORM_ROOT, "--no-log"] + extra_args
    return dataset_builder.main(args)


def _stem_files(tmp_path) -> list[str]:
    return sorted(p for p in os.listdir(tmp_path) if p.startswith("img_"))


def test_builder_single_curve_eval_ready(tmp_path):
    rc = _run_builder(tmp_path, ["--num-curves", "1"])
    assert rc == 0

    stems = [p[:-4] for p in _stem_files(tmp_path) if p.endswith(".png")
             and not p.endswith("_mask.png")]
    assert len(stems) == 3, stems

    for stem in stems:
        # sidecar contract
        for suffix in ("_mask.png", ".csv", "_meta.json", "_labels.json",
                       "_mcg.json", "_curves.json"):
            assert os.path.exists(os.path.join(tmp_path, stem + suffix)), suffix

        meta = json.load(open(os.path.join(tmp_path, stem + "_meta.json"),
                              encoding="utf-8"))
        assert meta["x_kind"] in ("linear", "log")
        assert meta["y_kind"] in ("linear", "log")
        assert len(meta["x_tick_values"]) >= 3
        assert len(meta["y_tick_values"]) >= 3
        assert meta["x_range"][0] < meta["x_range"][1]
        assert meta["y_range"][0] < meta["y_range"][1]
        assert meta["image_size"] == [meta["image_size"][0], meta["image_size"][1]]
        assert meta["num_curves"] == 1

        # GT csv: header + parseable rows
        rows = list(csv.reader(open(os.path.join(tmp_path, stem + ".csv"),
                                    encoding="utf-8")))
        assert rows[0] == ["x", "y"]
        data = np.array([[float(r[0]), float(r[1])] for r in rows[1:]])
        assert len(data) >= 50
        assert np.all(np.isfinite(data))

        # labels.json: anchored boxes + parseable tick texts
        labels = json.load(open(os.path.join(tmp_path, stem + "_labels.json"),
                                encoding="utf-8"))
        assert len(labels) >= 3
        for lb in labels:
            assert len(lb["box"]) == 4
            assert float(lb["text"]) is not None  # parseable number

        # mask: curve foreground present
        mask = np.asarray(__import__("cv2").imread(
            os.path.join(tmp_path, stem + "_mask.png"), 0))
        assert mask.shape[0] > 0 and mask.shape[1] > 0
        assert mask.max() == 255 and mask.sum() > 1000

        # MCG-JSON: platform schema + quality check passed
        mcg = json.load(open(os.path.join(tmp_path, stem + "_mcg.json"),
                             encoding="utf-8"))
        for key in ("dataset_info", "image", "plot_area", "axis", "style",
                    "legend", "curves", "quality_check"):
            assert key in mcg
        assert len(mcg["curves"]) == 1
        assert mcg["curves"][0]["data_points"]
        assert mcg["quality_check"]["passed"] is True


def test_builder_multi_curve_manifest(tmp_path):
    rc = _run_builder(tmp_path, ["--num-curves", "3"])
    assert rc == 0

    stems = [p[:-4] for p in _stem_files(tmp_path) if p.endswith(".png")
             and not p.endswith("_mask.png")]
    assert len(stems) == 3

    for stem in stems:
        meta = json.load(open(os.path.join(tmp_path, stem + "_meta.json"),
                              encoding="utf-8"))
        assert meta["num_curves"] == 3
        assert len(meta["curves_px"]) == 3
        # no single-curve csv for multi-curve samples (evaluate.py must fail loudly)
        assert not os.path.exists(os.path.join(tmp_path, stem + ".csv"))

        manifest = json.load(open(os.path.join(tmp_path, stem + "_curves.json"),
                                  encoding="utf-8"))
        assert manifest["num_curves"] == 3
        for entry in manifest["curves"]:
            csv_path = os.path.join(tmp_path, entry["csv"])
            assert os.path.exists(csv_path)
            rows = list(csv.reader(open(csv_path, encoding="utf-8")))
            assert rows[0] == ["x", "y"]

        mcg = json.load(open(os.path.join(tmp_path, stem + "_mcg.json"),
                             encoding="utf-8"))
        assert len(mcg["curves"]) == 3
        assert mcg["quality_check"]["passed"] is True
