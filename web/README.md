# 材料曲线工作台

提取与数据合成合一的本地 Web：`python web/app.py`（默认 http://127.0.0.1:8000，自动开浏览器）。

```powershell
conda activate mci
cd baseline
python web/app.py
# 可选：--port 8010 --no-browser
```

- **曲线提取**：上传 PNG/JPG（可批量）或点示例图 → 选 OCR / 分割模型 → 下载 CSV、JSON、叠加图、重绘图。
- **数据合成**：配置曲线类型、轴、DPI、线型、退化 → 后台生成 → 预览并下载 ZIP。

`web/synthesis_app.py` 仍可用，实际启动的是同一应用。任务产物在 `web/runs/` 与 `web/synthesis_runs/`（gitignore，启动时清理提取任务）。
