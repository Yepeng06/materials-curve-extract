from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import ExtractionConfig
from app.coordinate import pixel_to_data_points
from app.export import export_csv, export_json
from app.visualize import draw_overlay, draw_redrawn_curve


def _generate_mock_pixel_points(config: ExtractionConfig, n: int = 50) -> list[tuple[float, float]]:
    left, right = config.plot_area.left, config.plot_area.right
    top, bottom = config.plot_area.top, config.plot_area.bottom
    xs = np.linspace(left, right, n)
    mid = (top + bottom) / 2.0
    amp = max((bottom - top) * 0.25, 1.0)
    ys = mid + amp * np.sin(np.linspace(0, np.pi, n))
    ys = np.clip(ys, top, bottom)
    return list(zip(xs.tolist(), ys.tolist()))


def run_extraction(config: ExtractionConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(config.input_path))
    if image is None:
        raise ValueError(f"failed to read image: {config.input_path}")

    pixel_points = _generate_mock_pixel_points(config, n=max(config.resample_n // 16, 10))
    mapped_points = pixel_to_data_points(pixel_points, config.plot_area, config.x_range, config.y_range)

    csv_path = output_dir / "output.csv"
    export_csv(mapped_points, csv_path)

    overlay_path = output_dir / "extracted_overlay.png"
    draw_overlay(config.input_path, mapped_points, overlay_path)

    redrawn_path = output_dir / "redrawn_curve.png"
    draw_redrawn_curve(csv_path=csv_path, output_path=redrawn_path)

    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# V0 Report",
                "",
                "- 当前阶段为 V0 coordinate mapping implemented",
                f"- 本次输出点数: {len(mapped_points)}",
                f"- plot_area: {config.plot_area.model_dump()}",
                f"- x_range: {config.x_range.model_dump()}",
                f"- y_range: {config.y_range.model_dump()}",
                "- 当前曲线像素点仍是模拟点，真实图像提取将在下一任务实现",
            ]
        ),
        encoding="utf-8",
    )

    result = {
        "input_path": str(config.input_path),
        "output_dir": str(output_dir),
        "plot_area": config.plot_area.model_dump(),
        "x_range": config.x_range.model_dump(),
        "y_range": config.y_range.model_dump(),
        "mode": config.mode,
        "point_count": len(mapped_points),
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_files": {
            "output_csv": str(csv_path),
            "output_json": str(output_dir / "output.json"),
            "report_md": str(report_path),
            "extracted_overlay": str(overlay_path),
            "redrawn_curve": str(redrawn_path),
        },
    }

    json_path = output_dir / "output.json"
    export_json(result, json_path)
    return result
