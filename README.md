# mci — materials-curve-intel

材料曲线图智能提取 baseline：图像 → 结构化曲线数据（单曲线/单子图，多曲线接口已预留）。

## Web 工作台

提取与数据合成合一：

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
```

## 快速开始

```bash
conda activate mci
python scripts/gen_synthetic.py --out-dir data/synthetic --count 40 --seed 20260806
python scripts/run_baseline.py --image data/synthetic/img_0001.png --out-dir data/outputs --ocr stub --segmenter unet --debug
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval --ocr stub --segmenter unet
python -m pytest tests -q
```

训练 U-Net：`python train/train_segmentation.py --data-dir data/train_synthetic --val-dir data/synthetic --epochs 40 --batch 16 --out models/checkpoints/unet_curve.pt`

平台适配数据：`python data/dataset_builder.py --out-dir data/train_platform --count 2000 --seed 20260815 --yolo`

## 流程

结构检测 → 刻度 OCR → 线性/对数坐标映射 → 曲线分割（U-Net/CV）→ 骨架追踪 → CSV/JSON 导出。

验收：相对 RMSE ≤ 满量程 1%。Phase A 混合训练 U-Net（512）在平台单曲线集达标率 100%、合成集中位 0.40%。

## 已知限制

- 默认白底深色墨迹、左 y / 底 x；深底与右轴/顶轴未覆盖。
- 坐标类型判别需要 ≥3 个有效刻度。
- CPU OCR 约 6–15 s/图。
