from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class PlotArea(BaseModel):
    left: int
    top: int
    right: int
    bottom: int

    @model_validator(mode="after")
    def validate_box(self) -> "PlotArea":
        if self.left < 0 or self.top < 0:
            raise ValueError("plot_area left/top must be >= 0")
        if self.right <= self.left:
            raise ValueError("plot_area right must be greater than left")
        if self.bottom <= self.top:
            raise ValueError("plot_area bottom must be greater than top")
        return self


class AxisRange(BaseModel):
    min: float
    max: float

    @model_validator(mode="after")
    def validate_range(self) -> "AxisRange":
        if self.max <= self.min:
            raise ValueError("axis max must be greater than min")
        return self


class ExtractionConfig(BaseModel):
    input_path: Path
    output_dir: Path
    plot_area: PlotArea
    x_range: AxisRange
    y_range: AxisRange
    mode: Literal["gray", "hsv"] = "gray"
    hsv_lower: Optional[Tuple[int, int, int]] = Field(default=None)
    hsv_upper: Optional[Tuple[int, int, int]] = Field(default=None)
    resample_n: Optional[int] = Field(default=512, ge=2)

    @model_validator(mode="after")
    def validate_mode_related(self) -> "ExtractionConfig":
        if self.mode == "hsv" and (self.hsv_lower is None or self.hsv_upper is None):
            raise ValueError("hsv mode requires hsv_lower and hsv_upper")
        return self
