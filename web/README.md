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

1. **上传图片**：点击/拖拽上传 PNG/JPG（≤20MB），**支持一次多选/批量拖拽**；
   或直接点下方"示例图"一键测试（内置 6 张代表性示例图，无需 data/ 目录）；
2. **选择参数**：
   - 刻度 OCR：快速（示例图用标准答案刻度；用户上传图自动切换真实 OCR）/
     真实识别（PaddleOCR，CPU 单图约 6-15s）；
   - 曲线分割：深度学习 U-Net（推荐）/ 多曲线 U-Net（多曲线图）/ 经典 CV；
3. **点击"批量提取"**：逐张串行处理，文件列表实时显示状态
   （待提取/提取中/完成/失败），每张完成立即追加结果卡片；
4. **结果**：每张图一张卡片——overlay 叠加图（原图 + 提取曲线）+ 摘要
   （曲线数/点数/坐标类型/耗时/警告）；
5. **下载**：每张卡片独立提供 CSV 数据 / JSON 结构化结果 / 叠加图 PNG；
   "清空结果"可重置。

## 说明与限制（展示版）

- 提取结果与任务文件保存在 `web/runs/`（启动时清空，最多保留 300 个任务）；
- 内置示例图在 `web/examples/`（PNG + labels.json 侧车，已入库），数据目录
  （`data/`，gitignore）存在时示例区会额外列出其图片；
- 单曲线管线默认；多曲线图可选「多曲线 U-Net」segmenter（Phase C，多曲线实例分割）；
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
