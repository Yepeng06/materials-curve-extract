from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.config import AxisRange, ExtractionConfig, PlotArea
from app.pipeline import run_extraction

app = FastAPI(title="Materials Curve Extract V0")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")


def _parse_hsv(mode: str, lower: tuple[int | None, int | None, int | None], upper: tuple[int | None, int | None, int | None]):
    if mode == "gray":
        return None, None
    if None in lower or None in upper:
        raise ValueError("hsv 模式下必须提供完整 HSV 阈值")
    return tuple(int(x) for x in lower), tuple(int(x) for x in upper)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.post("/extract", response_class=HTMLResponse)
async def extract_web(
    request: Request,
    image: UploadFile = File(...),
    left: int = Form(...),
    top: int = Form(...),
    right: int = Form(...),
    bottom: int = Form(...),
    x_min: float = Form(...),
    x_max: float = Form(...),
    y_min: float = Form(...),
    y_max: float = Form(...),
    mode: str = Form("gray"),
    hsv_lower_h: int | None = Form(None),
    hsv_lower_s: int | None = Form(None),
    hsv_lower_v: int | None = Form(None),
    hsv_upper_h: int | None = Form(None),
    hsv_upper_s: int | None = Form(None),
    hsv_upper_v: int | None = Form(None),
    resample_n: int | None = Form(512),
):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("outputs") / run_id
    input_path = output_dir / image.filename
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(await image.read())

    try:
        hsv_lower, hsv_upper = _parse_hsv(
            mode,
            (hsv_lower_h, hsv_lower_s, hsv_lower_v),
            (hsv_upper_h, hsv_upper_s, hsv_upper_v),
        )
        cfg = ExtractionConfig(
            input_path=input_path,
            output_dir=output_dir,
            plot_area=PlotArea(left=left, top=top, right=right, bottom=bottom),
            x_range=AxisRange(min=x_min, max=x_max),
            y_range=AxisRange(min=y_min, max=y_max),
            mode=mode,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            resample_n=resample_n,
        )
        result = run_extraction(cfg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={"request": request, "result": result},
    )


@app.post("/api/extract")
async def extract_api(
    image: UploadFile = File(...),
    left: int = Form(...),
    top: int = Form(...),
    right: int = Form(...),
    bottom: int = Form(...),
    x_min: float = Form(...),
    x_max: float = Form(...),
    y_min: float = Form(...),
    y_max: float = Form(...),
    mode: str = Form("gray"),
    hsv_lower_h: int | None = Form(None),
    hsv_lower_s: int | None = Form(None),
    hsv_lower_v: int | None = Form(None),
    hsv_upper_h: int | None = Form(None),
    hsv_upper_s: int | None = Form(None),
    hsv_upper_v: int | None = Form(None),
    resample_n: int | None = Form(512),
):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("outputs") / run_id
    input_path = output_dir / image.filename
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(await image.read())

    try:
        hsv_lower, hsv_upper = _parse_hsv(
            mode,
            (hsv_lower_h, hsv_lower_s, hsv_lower_v),
            (hsv_upper_h, hsv_upper_s, hsv_upper_v),
        )
        cfg = ExtractionConfig(
            input_path=input_path,
            output_dir=output_dir,
            plot_area=PlotArea(left=left, top=top, right=right, bottom=bottom),
            x_range=AxisRange(min=x_min, max=x_max),
            y_range=AxisRange(min=y_min, max=y_max),
            mode=mode,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            resample_n=resample_n,
        )
        return run_extraction(cfg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
