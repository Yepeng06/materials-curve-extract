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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("outputs") / run_id
    input_path = output_dir / image.filename
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(await image.read())

    try:
        cfg = ExtractionConfig(
            input_path=input_path,
            output_dir=output_dir,
            plot_area=PlotArea(left=left, top=top, right=right, bottom=bottom),
            x_range=AxisRange(min=x_min, max=x_max),
            y_range=AxisRange(min=y_min, max=y_max),
            mode=mode,
        )
        result = run_extraction(cfg)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return templates.TemplateResponse("result.html", {"request": request, "result": result})


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
):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path("outputs") / run_id
    input_path = output_dir / image.filename
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(await image.read())

    cfg = ExtractionConfig(
        input_path=input_path,
        output_dir=output_dir,
        plot_area=PlotArea(left=left, top=top, right=right, bottom=bottom),
        x_range=AxisRange(min=x_min, max=x_max),
        y_range=AxisRange(min=y_min, max=y_max),
        mode=mode,
    )
    return run_extraction(cfg)
