# 材料科学图像曲线识别与智能解析（V0-Midterm）

## 项目目标
构建一个 **Web 优先** 的端到端基线：用户上传曲线图，手动输入绘图区与坐标范围，系统输出可下载结果文件。

## V0 范围
- FastAPI + Jinja2 网页
- `/extract` 网页提取入口，`/api/extract` API 入口
- 统一调用 `run_extraction()` pipeline
- 输出 CSV / JSON / mask / overlay / redrawn / report

## 安装依赖
```bash
pip install -r requirements.txt
```

## 本地启动网页
```bash
uvicorn app.main:app --reload
```

浏览器访问：
- http://127.0.0.1:8000

## 当前限制
- V0 为 scaffold，占位算法为主
- 不包含深度学习、OCR、图例匹配、多曲线自动分离
