import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.config import AxisRange, ExtractionConfig, PlotArea
from app.pipeline import run_extraction


def test_pipeline_smoke(tmp_path: Path):
    img = np.full((500, 700, 3), 255, dtype=np.uint8)
    cv2.line(img, (120, 420), (620, 80), (0, 0, 0), 2)
    input_path = tmp_path / "input.png"
    cv2.imwrite(str(input_path), img)

    out_dir = tmp_path / "out"
    cfg = ExtractionConfig(
        input_path=input_path,
        output_dir=out_dir,
        plot_area=PlotArea(left=100, top=50, right=650, bottom=450),
        x_range=AxisRange(min=0, max=1000),
        y_range=AxisRange(min=0, max=10),
        mode="gray",
        resample_n=256,
    )
    run_extraction(cfg)

    assert (out_dir / "output.csv").exists()
    assert (out_dir / "output.json").exists()
    assert (out_dir / "extracted_overlay.png").exists()
    assert (out_dir / "redrawn_curve.png").exists()
    assert (out_dir / "report.md").exists()

    df = pd.read_csv(out_dir / "output.csv")
    assert len(df) >= 10
    assert list(df.columns) == ["index", "pixel_x", "pixel_y", "x", "y"]
    assert df["x"].notna().all()
    assert df["y"].notna().all()

    meta = json.loads((out_dir / "output.json").read_text(encoding="utf-8"))
    assert meta["point_count"] == len(df)
