# Win11 PowerShell 运行步骤

```powershell
cd <repo_path>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

浏览器打开：`http://127.0.0.1:8000`

运行测试：
```powershell
pytest
```
