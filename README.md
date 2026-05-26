# 材料科学图像曲线识别与智能解析（V0 OpenCV Baseline）

## 项目目标
构建 **Web 优先** 端到端基线：上传曲线图 + 手工输入坐标信息，输出结构化曲线数据。

## V0 当前能力
- FastAPI + Jinja2 Web 入口，`/extract` 与 `/api/extract` 共用 `run_extraction()`。
- 手工输入 `plot_area`、`x_range`、`y_range`。
- OpenCV 曲线提取：`gray`（深色曲线）/ `hsv`（彩色曲线）两种模式。
- 输出 `output.csv`、`output.json`、`curve_mask.png`、`extracted_overlay.png`、`redrawn_curve.png`、`report.md`。

## 当前限制
- 适合清晰单曲线与简单彩色单曲线。
- 暂不保证复杂多曲线/交叉/严重噪声。
- 不做自动 OCR、自动 plot_area 检测、深度学习模型。
- HSV 阈值需手工输入。

## Win11 本地运行
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --reload
```

浏览器访问： http://127.0.0.1:8000

## Python 调用示例
```python
from pathlib import Path
from app.config import AxisRange, PlotArea, ExtractionConfig
from app.pipeline import run_extraction

cfg = ExtractionConfig(
    input_path=Path("demo.png"),
    output_dir=Path("outputs/demo_run"),
    plot_area=PlotArea(left=100, top=80, right=900, bottom=700),
    x_range=AxisRange(min=0, max=1000),
    y_range=AxisRange(min=0, max=10),
    mode="gray",
    resample_n=512,
)
result = run_extraction(cfg)
print(result["output_files"]["output_csv"])
```
