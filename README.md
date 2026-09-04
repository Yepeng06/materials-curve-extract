# mci — materials-curve-intel

材料曲线图智能提取 baseline：图像 → 结构化曲线数据（单曲线/多曲线，模型自动判别）。

## Web 工作台

提取与数据合成一体化的全屏工作台，左侧导航切换 **曲线提取 / 数据合成**：

- **上传或示例图提取**：内置/合成/平台三类示例池可"换一批"随机刷新；
- **提取参数**：分割模型 `自动判断`（多通道 U-Net 计数 → 自动选单曲线/多曲线/CV），导出点数 4/16/32/64/128/原生（仅作用于 CSV 与汇总统计，result.json 恒为原生密度）；
- **人工校正**：每张结果卡片内嵌 Konva 校正编辑器（拖拽锚点 / Shift+点击加点 / Delete 删点），保存后服务端像素→数据反算、重写 result.json、重导出 CSV/叠加图/重绘图，并落库修正审计轨迹（`corrections.json` + `meta.corrections`）；
- **刻度 OCR**：默认 PP-OCRv5 server（打印/手写小字显著提升），自动回退 v4 mobile；上标碎片自动归位（`10⁻³` → `10^-3`），数值解析支持 ±/~//>/单位后缀（MPa、°C…）/易混字形（O→0、l→1）。

```powershell
conda activate mci
cd baseline
python web/app.py    # http://127.0.0.1:8000
```

详见 `web/README.md`。

## 环境

独立 conda 环境 **`mci`**（Python 3.11）。Windows 上 torch 与 paddlepaddle-gpu CUDA DLL 冲突，OCR 用 CPU Paddle，U-Net 用 GPU。

```bash
conda create -n mci python=3.11 -y
conda activate mci
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install ultralytics==8.4.115 paddlepaddle==3.3.1 paddleocr==3.7.0 \
    opencv-python scikit-image scipy matplotlib pandas pyyaml pytest tqdm "numpy==1.26.4"
python -m pip install "fastapi>=0.110" "uvicorn>=0.29" "python-multipart>=0.0.9"
```

## 快速开始

```bash
conda activate mci
python scripts/gen_synthetic.py --out-dir data/synthetic --count 40 --seed 20260806
python scripts/run_baseline.py --image data/synthetic/img_0001.png --out-dir data/outputs --ocr stub --segmenter auto --debug
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval --ocr stub --segmenter unet
python scripts/eval_tick_ocr.py --dir data/synthetic --tier server --limit 20   # 刻度 OCR 准确率评测
python -m pytest tests -q
```

训练 U-Net：`python train/train_segmentation.py --data-dir data/train_synthetic --val-dir data/synthetic --epochs 40 --batch 16 --out models/checkpoints/unet_curve.pt`

平台适配数据：`python data/dataset_builder.py --out-dir data/train_platform --count 2000 --seed 20260815 --yolo`

## 流程

结构检测 → 刻度 OCR（v5 server，上标归位）→ 线性/对数坐标映射 → 曲线分割（`auto`：多通道 U-Net 计数自动选路 / 单曲线 U-Net / 多曲线 U-Net / CV）→ 骨架追踪 → 点数重采样导出 → CSV/JSON。

验收：相对 RMSE ≤ 满量程 1%。Phase A 混合训练 U-Net（512）在平台单曲线集达标率 100%、合成集中位 0.40%。

## 已知限制

- 默认白底深色墨迹、左 y / 底 x；深底与右轴/顶轴未覆盖。
- 坐标类型判别需要 ≥3 个有效刻度。
- CPU OCR 约 6–15 s/图（v5 server 较 v4 mobile 更慢但更准；`MCI_OCR_TIER=mobile` 可切回）。
- `tests/test_pipeline.py::test_end_to_end_linear` 在 seed 11 + CV 后端存在既有波动（与本次改动无关）。
