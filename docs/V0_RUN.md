# V0 运行说明

1. 安装依赖：`pip install -r requirements.txt`
2. 运行测试：`pytest -q`
3. 启动服务：`uvicorn app.main:app --reload`
4. 打开：`http://127.0.0.1:8000`

Web/API 参数：
- `plot_area`: left/top/right/bottom
- `x_range`: x_min/x_max
- `y_range`: y_min/y_max
- `mode`: gray 或 hsv
- `hsv_lower`/`hsv_upper`: hsv 模式必填
- `resample_n`: 可选
