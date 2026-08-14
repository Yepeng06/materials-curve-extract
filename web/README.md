# Web 演示系统（展示版）

材料曲线智能提取的本地 Web 演示：上传曲线图 → 一键提取 → overlay 叠加图
展示 → 下载结构化数据（CSV/JSON）与叠加图。

## 启动

```powershell
conda activate mci
cd F:\CODE\New\baseline
python web/app.py
```

- 默认地址 **http://127.0.0.1:8000**，启动后**自动打开浏览器**；
- 可选参数：`--port 8010` 换端口；`--no-browser` 不自动开浏览器；
- 首次启动后台预热 PaddleOCR（10-30s，终端会打印"PaddleOCR 预热完成"），
  预热期间不影响页面使用。

## 使用

1. **上传图片**：点击/拖拽上传 PNG/JPG（≤20MB），或直接点下方"示例图"一键测试；
2. **选择参数**：
   - 刻度 OCR：快速（示例图用标准答案刻度；用户上传图自动切换真实 OCR）/
     真实识别（PaddleOCR，单图约 6s CPU）；
   - 曲线分割：深度学习 U-Net（推荐）/ 经典 CV；
3. **点击"提取"**：约 0.5-2s（stub）或 6-15s（真实 OCR）；
4. **结果**：overlay 叠加图（原图 + 提取曲线）+ 摘要表（曲线数/点数/坐标类型/
   耗时/警告）；
5. **下载**：CSV 数据 / JSON 结构化结果 / 叠加图 PNG。

## 说明与限制（展示版）

- 提取结果与任务文件保存在 `web/runs/`（启动时清空，最多保留 300 个任务）；
- 当前为单曲线管线：多曲线图只输出主曲线（多曲线提取属 Phase C，尚未接入）；
- 真实 OCR 用 CPU 版 PaddleOCR（Windows 下与 torch 的 DLL 冲突，已按
  `tick_reader.py` 的先 torch 后 paddle 顺序规避）；GPU OCR 属 Phase 2 计划；
- stub 模式依赖数据集的 GT 侧车（`*_labels.json`），仅示例图可用；
  用户上传图即使选择"快速"也会自动切换真实 OCR 并在页面提示。

## 文件

```
web/
├── app.py              # FastAPI 后端（提取/示例/下载端点，自动开浏览器）
├── templates/index.html
├── static/style.css    # 前端（原生 HTML/CSS/JS，无构建）
├── static/app.js
└── runs/               # 运行期任务产物（gitignore）
```
