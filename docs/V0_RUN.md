# V0 运行说明（Win11）

```powershell
cd <repo_path>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
uvicorn app.main:app --reload
```

浏览器打开：`http://127.0.0.1:8000`

## 说明
- 当前 V0 已完成 Web 骨架 + 坐标映射核心。
- 当前曲线点为模拟点，仅用于验证参数链路和坐标映射。
- 下一阶段将实现 OpenCV 真实曲线像素提取。
- 若 Codex 环境无法联网安装依赖，请在本地 Win11 环境运行上述命令。
