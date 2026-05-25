# 材料科学图像曲线识别与智能解析（V0-Midterm）

## 项目目标
构建一个 **Web 优先** 的端到端基线：用户上传曲线图，手动输入绘图区与坐标范围，系统输出可下载结果文件。

## V0 当前完成
- FastAPI + Jinja2 网页骨架
- `/extract` 网页提取入口，`/api/extract` API 入口
- Web 与 API 统一调用 `run_extraction()` pipeline
- 坐标映射核心：pixel → data 线性映射（含 y 轴反向）
- 输出 CSV / JSON / extracted_overlay / redrawn_curve / report

## 当前限制
- V0 不包含深度学习、OCR、YOLO、U-Net
- 当前曲线像素点仍是模拟点
- 下一阶段将接入 OpenCV 真实曲线像素提取

## Win11 本地运行
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --reload
```

浏览器访问：
- http://127.0.0.1:8000

## Codex 环境依赖说明
若在线环境 `pip install -r requirements.txt` 因网络或源限制失败（如 403），请在本地 Win11 环境完成依赖安装后执行测试。
