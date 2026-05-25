# V0 设计说明（Web 优先）

## 架构
- Web UI：`templates/index.html` + `templates/result.html`
- FastAPI Routes：`GET /`，`POST /extract`，`POST /api/extract`
- Pipeline：`app/pipeline.py::run_extraction()`
- 输出目录：`outputs/<run_id>/`

## 关键原则
1. Web 与 API 共用同一 pipeline，避免逻辑分叉。
2. 坐标映射统一在 `app/coordinate.py` 实现，避免重复算法。
3. 当前曲线像素点使用模拟点，先打通“输入参数→映射→导出→可视化”链路。
4. V0 阶段不引入深度学习/OCR。

## V0 本次新增核心
- 线性映射公式（包含 y 反向）
- 单点/批量映射接口
- CSV 固定列：`index,pixel_x,pixel_y,x,y`
- output.json 元数据链路
- overlay 与 redrawn 可视化用于人工核验
