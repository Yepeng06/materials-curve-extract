# Agent Rules

- V0 阶段不做深度学习/OCR。
- Web 和 API 都必须调用同一个 `run_extraction` pipeline。
- 每次修改后运行 `pytest`。
- 不提交 `outputs/` 下生成文件，仅保留 `outputs/.gitkeep`。
