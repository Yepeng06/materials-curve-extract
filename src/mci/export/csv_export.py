"""CSV export: one file per curve, x/y data coordinates."""
from __future__ import annotations

import os

from ..schema import ExtractionResult


def write_csv(result: ExtractionResult, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    lines = [
        f"# mci baseline {__import__('mci').__version__}",
        f"# source: {result.image_path}",
        f"# x_axis: kind={result.x_axis.kind.value} range=[{result.x_axis.vmin:.6g}, {result.x_axis.vmax:.6g}] quality={result.x_axis.quality:.4f}",
        f"# y_axis: kind={result.y_axis.kind.value} range=[{result.y_axis.vmin:.6g}, {result.y_axis.vmax:.6g}] quality={result.y_axis.quality:.4f}",
        "x,y",
    ]
    for curve in result.curves:
        if curve.legend_label:
            lines.append(f"# curve: {curve.name} label={curve.legend_label} color={curve.color}")
        for x, y in curve.points:
            lines.append(f"{x:.8g},{y:.8g}")
    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path
