# 材料曲线工作台

提取、人工校正与数据合成一体化的全屏本地 Web（左侧导航，无外部 CDN，离线可用）：

```powershell
conda activate mci
cd baseline
python web/app.py
# 可选：--port 8010 --no-browser
```

## 曲线提取

- **上传** PNG/JPG（可批量拖拽）或点击**示例图**（`换一批` 随机刷新示例池：内置示例 / 合成测试集 / 平台模板集）。
- **提取参数**
  - 分割模型：`自动判断`（默认）——多通道 U-Net 输出通道数即曲线计数：≥2 条走多曲线、1 条走单曲线 U-Net 精化、全失败回退经典 CV；也可手动指定 `multi_unet` / `unet` / `cv`。
  - 导出点数：4 / 16 / 32 / 64 / 128 / 原生（默认 128）。只作用于 CSV 与汇总统计，`result.json` 恒为原生密度（评估协议与导出解耦）；对数轴在对数空间等距插值。
  - 刻度 OCR：快速（示例图用标准答案）或真实识别（PP-OCRv5 server，自动回退 v4 mobile；上标碎片归位、单位/±/易混字形解析）。
- **结果卡片**：原图 / 重绘图 / 叠加图三联预览 + 汇总（自动判定徽标、导出点数、耗时）+ 全部产物下载。
- **人工校正**（卡片内嵌，Konva 本地化 `static/vendor/konva.min.js`）：
  - 拖拽锚点微调、`Shift+点击` 加点（按 x 有序插入）、点选后 `Delete`/按钮删点；
  - 曲线列表可切换激活曲线与显隐；
  - 保存 → `POST /api/runs/{tid}/correct`：服务端按存储的轴映射像素→数据反算，重写 `result.json`，重导出 CSV（沿用提取时的点数设置）/叠加图/重绘图；
  - 修正审计（goal.md E4）：`web/runs/<tid>/corrections.json`（JSONL，可回放）+ `result.json` 的 `meta.corrections`（moved/added/removed 操作计数）。

## 数据合成

配置曲线类型、坐标轴、曲线数、DPI、线型与退化 → 后台生成 → 实时进度 → 预览 + 下载 ZIP。

## 接口速览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/extract` | 上传图提取（`ocr` / `segmenter=auto` / `points`） |
| POST | `/api/extract_example` | 示例图提取（同上参数） |
| GET | `/api/examples?refresh=1` | 示例池（随机换一批） |
| GET | `/api/runs/{tid}/csv·json·overlay·redraw·image` | 任务产物 |
| POST | `/api/runs/{tid}/correct` | 人工校正（像素锚点 → 反算 → 重导出 + 审计） |
| POST | `/api/synthesize` | 数据合成任务 |

任务产物在 `web/runs/` 与 `web/synthesis_runs/`（gitignore，启动时清理提取任务）。`web/synthesis_app.py` 仍可用，实际启动的是同一应用。
