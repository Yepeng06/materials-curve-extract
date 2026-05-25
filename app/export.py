from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CSV_COLUMNS = ["index", "pixel_x", "pixel_y", "x", "y"]


def export_csv(data_points, output_path: Path) -> None:
    rows = []
    for idx, point in enumerate(data_points):
        rows.append(
            {
                "index": idx,
                "pixel_x": point["pixel_x"],
                "pixel_y": point["pixel_y"],
                "x": point["x"],
                "y": point["y"],
            }
        )
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8")


def export_json(metadata: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
