"""兼容入口：数据合成已并入一体化工作台 `web/app.py`。"""
from __future__ import annotations

import sys
from pathlib import Path

# 与 web/app.py 同进程加载，复用同一 FastAPI 应用。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from web.app import app, main  # noqa: E402, F401

if __name__ == "__main__":
    sys.exit(main())
