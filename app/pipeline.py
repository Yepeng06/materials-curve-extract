from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.config import ExtractionConfig
from app.coordinate import pixel_to_data_points
from app.export import export_csv, export_json
from app.extract_curve import extract_curve_mask
from app.preprocess import crop_plot_area, load_image, save_image
from app.reconstruct import mask_to_curve_points
from app.visualize import draw_overlay, draw_redrawn_curve


EXTRACTOR_VERSION = "v0-opencv-baseline"


def run_extraction(config: ExtractionConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(config.input_path)
    save_image(output_dir / "input.png", image)

    cropped = crop_plot_area(image, config.plot_area)
    save_image(output_dir / "cropped_plot_area.png", cropped)

    curve_mask = extract_curve_mask(
        cropped_image=cropped,
        mode=config.mode,
        hsv_lower=config.hsv_lower,
        hsv_upper=config.hsv_upper,
    )
    save_image(output_dir / "curve_mask.png", curve_mask)

    pixel_points = mask_to_curve_points(curve_mask, config.plot_area, resample_n=config.resample_n)
    mapped_points = pixel_to_data_points(pixel_points, config.plot_area, config.x_range, config.y_range)

    csv_path = output_dir / "output.csv"
    export_csv(mapped_points, csv_path)

    overlay_path = output_dir / "extracted_overlay.png"
    draw_overlay(config.input_path, mapped_points, overlay_path)

    redrawn_path = output_dir / "redrawn_curve.png"
    draw_redrawn_curve(data_points=mapped_points, output_path=redrawn_path)

    report_path = output_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# V0 OpenCV Baseline Report",
                "",
                "- 当前阶段：V0 OpenCV baseline（真实曲线像素提取）",
                f"- 提取模式: {config.mode}",
                f"- 提取点数: {len(mapped_points)}",
                f"- plot_area: {config.plot_area.model_dump()}",
                f"- x_range: {config.x_range.model_dump()}",
                f"- y_range: {config.y_range.model_dump()}",
                "- 当前限制：适合清晰单曲线，不适合复杂多曲线、严重噪声、自动 OCR。",
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
        "hsv_lower": list(config.hsv_lower) if config.hsv_lower is not None else None,
        "hsv_upper": list(config.hsv_upper) if config.hsv_upper is not None else None,
        "resample_n": config.resample_n,
        "point_count": len(mapped_points),
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extractor_version": EXTRACTOR_VERSION,
        "output_files": {
            "input_png": str(output_dir / "input.png"),
            "cropped_plot_area": str(output_dir / "cropped_plot_area.png"),
            "curve_mask": str(output_dir / "curve_mask.png"),
            "output_csv": str(csv_path),
            "output_json": str(output_dir / "output.json"),
            "report_md": str(report_path),
            "extracted_overlay": str(overlay_path),
            "redrawn_curve": str(redrawn_path),
        },
    }

    export_json(result, output_dir / "output.json")
    return result
