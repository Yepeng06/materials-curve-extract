from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.config import ExtractionConfig
from app.coordinate import pixel_to_data
from app.export import export_csv
from app.extract_curve import extract_curve_mask
from app.preprocess import crop_plot_area, read_image
from app.reconstruct import reconstruct_curve_points
from app.visualize import save_mask, save_overlay, save_redrawn


def run_extraction(config: ExtractionConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(config.input_path)
    input_saved = output_dir / "input.png"
    cv2.imwrite(str(input_saved), image)

    cropped = crop_plot_area(image, config.plot_area)
    cropped_path = output_dir / "cropped_plot_area.png"
    cv2.imwrite(str(cropped_path), cropped)

    mask = extract_curve_mask(cropped, mode=config.mode)
    mask_path = output_dir / "curve_mask.png"
    save_mask(mask, mask_path)

    overlay_path = output_dir / "extracted_overlay.png"
    save_overlay(cropped, mask, overlay_path)

    redrawn_path = output_dir / "redrawn_curve.png"
    save_redrawn(mask, redrawn_path)

    points_px = reconstruct_curve_points(mask, resample_n=config.resample_n or 512)
    points_data = pixel_to_data(
        points_px,
        width=max(cropped.shape[1], 2),
        height=max(cropped.shape[0], 2),
        x_range=config.x_range,
        y_range=config.y_range,
    )

    csv_path = output_dir / "output.csv"
    export_csv(points_data, csv_path)

    result = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "input_path": str(config.input_path),
            "plot_area": config.plot_area.model_dump(),
            "x_range": config.x_range.model_dump(),
            "y_range": config.y_range.model_dump(),
            "mode": config.mode,
        },
        "outputs": {
            "input": str(input_saved),
            "cropped_plot_area": str(cropped_path),
            "curve_mask": str(mask_path),
            "extracted_overlay": str(overlay_path),
            "redrawn_curve": str(redrawn_path),
            "output_csv": str(csv_path),
        },
    }

    json_path = output_dir / "output.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(
        "# V0 Scaffold Report\n\nThis is a V0 scaffold run. Curve extraction/reconstruction algorithms are placeholders.\n",
        encoding="utf-8",
    )

    result["outputs"]["output_json"] = str(json_path)
    result["outputs"]["report_md"] = str(report_path)
    return result
