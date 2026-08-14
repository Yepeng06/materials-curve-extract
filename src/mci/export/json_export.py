"""JSON export: full structured extraction result (schema-complete)."""
from __future__ import annotations

import json
import os

from ..schema import ExtractionResult


def _axis_to_dict(axis):
    return {
        "role": axis.role.value,
        "kind": axis.kind.value,
        "slope": axis.slope,
        "intercept": axis.intercept,
        "vmin": axis.vmin,
        "vmax": axis.vmax,
        "pmin": axis.pmin,
        "pmax": axis.pmax,
        "quality": axis.quality,
        "ticks": [
            {"pixel": t.pixel, "value": t.value, "text": t.text, "score": t.score}
            for t in axis.ticks
        ],
    }


def write_json(result: ExtractionResult, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    data = {
        "image_path": result.image_path,
        "x_axis": _axis_to_dict(result.x_axis),
        "y_axis": _axis_to_dict(result.y_axis),
        "plot_bbox": list(result.structure.plot_bbox),
        "curves": [
            {
                "name": c.name,
                "color": list(c.color),
                "legend_label": c.legend_label,
                "points": [[x, y] for x, y in c.points],
            }
            for c in result.curves
        ],
        "warnings": result.warnings,
        "meta": result.meta,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path
