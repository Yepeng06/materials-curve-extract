# V0 设计说明（Web 优先）

## 架构
- Web UI：`templates/index.html` + `templates/result.html`
- FastAPI Routes：`GET /`，`POST /extract`，`POST /api/extract`
- Pipeline：`app/pipeline.py::run_extraction()`
- 输出目录：`outputs/<run_id>/`

## 关键原则
1. Web 与 API 共用同一 pipeline，避免逻辑分叉。
2. 算法模块化拆分，便于后续替换为真实实现。
3. 当前只保证可运行、可测试、可迭代。
