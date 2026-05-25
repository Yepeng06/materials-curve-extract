from pathlib import Path

import cv2
import numpy as np

from app.config import AxisRange, ExtractionConfig, PlotArea
from app.pipeline import run_extraction


def test_pipeline_smoke(tmp_path: Path):
    img = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.line(img, (10, 100), (150, 20), (0, 0, 0), 2)
    input_path = tmp_path / "input.png"
    cv2.imwrite(str(input_path), img)

    out_dir = tmp_path / "out"
    cfg = ExtractionConfig(
        input_path=input_path,
        output_dir=out_dir,
        plot_area=PlotArea(left=0, top=0, right=160, bottom=120),
        x_range=AxisRange(min=0, max=10),
        y_range=AxisRange(min=0, max=5),
        mode="gray",
    )
    run_extraction(cfg)

    assert (out_dir / "output.csv").exists()
    assert (out_dir / "output.json").exists()
    assert (out_dir / "cropped_plot_area.png").exists()
    assert (out_dir / "report.md").exists()
