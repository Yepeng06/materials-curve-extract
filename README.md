# mci — materials-curve-intel 曲线提取 baseline

> 含展示版 Web 演示系统（`web/`），启动：`python web/app.py`（自动开浏览器，
> 详见 `web/README.md`）。

材料科学图像曲线智能识别与解析项目的 **Baseline 框架**（单曲线、单子图），
端到端实现「图像 → 结构化曲线数据」，并为多曲线/多子图扩展预留了接口。

## 1. 目标与范围（本 baseline 的定位）

| 能力 | 状态 |
|------|------|
| 单曲线、单子图曲线图提取（合成数据） | ✅ 端到端可用 |
| 论文风格单曲线图提取（高精度） | ✅ 端到端可用（U-Net 后端） |
| 线性 / 对数坐标自动判别 | ✅ 拟合残差比较（R² 判据） |
| 虚线 / 网格 / 浅色曲线 / JPEG 退化 | ✅ U-Net 后端鲁棒（CV 后端部分覆盖） |
| 多曲线提取 | 🔲 接口已预留（Curve 为列表 + legend_matcher 协议） |
| 多子图 | 🔲 Phase 2 |

验收指标（Prompt.md 第四节）：曲线点 RMSE ≤ 满量程 1%（当前合成集
U-Net 后端达标，见 §7）。

## 2. 环境（Anaconda）

已创建独立环境 **`mci`**（Python 3.11，RTX 4060 / CUDA 12.6）：

```bash
conda create -n mci python=3.11 -y
conda activate mci
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install ultralytics==8.4.115 paddlepaddle==3.3.1 paddleocr==3.7.0 \
    opencv-python scikit-image scipy matplotlib pandas pyyaml pytest tqdm "numpy==1.26.4"
```

> 注：Windows 上 torch 与 paddlepaddle-gpu 的 CUDA/cuDNN DLL 互相冲突（WinError 127），
> 因此 OCR 使用 **CPU paddle**（刻度标签量小，单图 ~6 s，稳态），U-Net 用 GPU（torch）。
> GPU OCR 可在 Phase 2 以独立推理服务实现。

## 3. 快速开始

```bash
conda activate mci

# 1) 生成合成测试集（PNG + GT CSV + 掩码 + meta）
python scripts/gen_synthetic.py --out-dir data/synthetic --count 40 --seed 20260806

# 2) 生成训练集（含曲线掩码标注 *_mask.png）
python scripts/gen_synthetic.py --out-dir data/train_synthetic --count 400 --seed 100

# 3) 训练 U-Net 曲线分割模型（GPU）
python train/train_segmentation.py --data-dir data/train_synthetic \
    --val-dir data/synthetic --epochs 40 --batch 16 \
    --out models/checkpoints/unet_curve.pt

# 4) 单图提取（OCR 可选 stub[确定性] / paddle[真实]）
python scripts/run_baseline.py --image data/synthetic/img_0001.png \
    --out-dir data/outputs --ocr stub --segmenter unet --debug

# 5) 批量评估（RMSE / 覆盖率 / 坐标类型）
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval \
    --ocr stub --segmenter unet

# 6) 单元测试
python -m pytest tests -q
```

## 4. 架构

```
baseline/
├── src/mci/
│   ├── schema.py                  # 统一数据模型（Axis/Tick/Curve/Result）
│   ├── utils.py                   # 图像 IO、自适应二值化、刻度数值解析
│   ├── pipeline/
│   │   ├── base.py                # 模块接口（OCR/检测/分割/图例匹配协议）
│   │   ├── chart_structure.py     # 绘图区/轴线/刻度线检测（经典 CV）
│   │   ├── tick_reader.py         # 刻度 OCR（PaddleOCR / Stub）+ 标签关联
│   │   ├── coordinate_mapper.py   # 像素→数据映射，线性/对数自动判别
│   │   ├── curve_extractor.py     # 曲线提取（骨架化+追踪+列中值回退）
│   │   ├── segmenter.py           # 分割后端：U-Net（可训练）/ CV 后备
│   │   ├── legend_matcher.py      # 图例匹配占位（多曲线 Phase 2）
│   │   └── extractor.py           # 端到端编排入口
│   ├── models/segmentation/unet.py# U-Net（PyTorch 原生，~2.1M 参数）
│   ├── export/                    # CSV / JSON / 可视化叠加
│   └── eval/metrics.py            # RMSE / 相对 RMSE / x 覆盖率
├── scripts/
│   ├── gen_synthetic.py           # 合成数据工厂（GT 精确到像素，含自检）
│   ├── run_baseline.py            # 单图/批量提取 CLI
│   └── evaluate.py                # 批量评估 + 报告
├── train/train_segmentation.py    # U-Net 训练脚本
├── configs/baseline.yaml          # 全部可调参数
└── tests/                         # 单元 + 端到端测试
```

流程：`结构检测 → 刻度 OCR → 坐标映射（线性/对数自动）→ 曲线分割（U-Net/CV）→
骨架化/追踪 → 像素→数据映射 → CSV/JSON 导出`。

### 4.1 坐标类型判别（线性 vs 对数）

对每条轴同时拟合 `value = a·p+b` 与 `log10(value) = a·p+b`（p 沿轴定向），
以 R² / 残差判据选择：

- log 胜出当且仅当 `R²_log > R²_lin + 0.01`，或（`R²_lin < 0.999` 且
  `RMS_log < 0.25·RMS_lin`）；
- 两者都完美（如仅 2 个刻度）时默认线性并在结果中给出 warning。

### 4.2 网格线去除（CV 后端）

两级机制，均对曲线免疫：

1. **轮廓扫描 + 刻度对齐**：网格线必过刻度位置；行/列墨迹覆盖率 > 45% 且
   与刻度 ±1px 对齐才删除（虚线网格同样命中，曲线厚度仅 1-4px 不可能达到 45%）；
2. **Hough 直线簇**：同一根线的线段先按截距聚类（防止把曲线直线段当多条线），
   仅删除「浅色平行簇」或「等间距 ≥3 簇（保留最暗簇）」或「刻度对齐」的线。

删除后以 (7,3)/(3,7) 闭运算桥接曲线在网格交叉处的 1-3px 缺口。

### 4.3 虚线曲线桥接（CV 后端）

虚线每段 dash 是独立连通域；在**组件选择之前**，对骨架端点做
「方向对齐 + 异组件 + 端点仅用一次 + 最短优先」的配对桥接（≤48px）。

## 5. 为什么需要训练模型（方法学依据）

纯 CV 阈值化在对抗场景（浅色曲线、虚线、网格、JPEG）达到上限：
二值化阈值、网格/组件启发式需要逐例调参。学术界一致做法是**合成数据驱动的
分割模型**：

- **APEX-Net** (arXiv:2101.06217)：深度网络 + 合成数据训练，自动提取绘图区与曲线；
- **ChartOCR** (WACV 2021)：检测/分割 + OCR 的混合框架（本文的 U-Net + PaddleOCR 结构同源）；
- **WebPlotDigitizer**：经典半自动工具，坐标校准与曲线追踪启发式（本 baseline 的
  刻度拟合/骨架追踪沿用其思路并全自动化）；
- **Graph-FINDER** (npj Comput. Mater. 2026)、**AI-ChartParser** (CGF 2025)：
  材料/论文图表数据提取的最新工程实践。

本 baseline 的 U-Net 用合成数据训练（`gen_synthetic.py` 输出精确到像素的
曲线掩码 GT，虚线也按实线绘制——让网络学会补全断线），推理时分割图只含曲线，
天然免疫网格/文字/图例，且**提取点数充足**（整条曲线连通后追踪，2000 点上限）。

## 6. 数据集与 Ground Truth

`scripts/gen_synthetic.py` 自包含合成数据工厂（生产数据源为
materials-curve-dataset-platform，见 Roadmap）：

- 5 类曲线（蠕变式/幂律/S 型/应力松弛/对数），线性/对数轴 4 组合；
- 论文风与实验风样式（衬线/网格/边框开关/图例内外/标题）；
- 退化：JPEG、缩放往返、模糊、亮度对比度、椒盐；
- 每图输出：`<stem>.png`（退化后图像）、`<stem>.csv`（GT 数据坐标）、
  `<stem>_mask.png`（曲线掩码）、`<stem>_meta.json`（轴类型/范围/刻度值/样式）、
  `<stem>_labels.json`（刻度标签框，stub OCR 用）；
- **GT 像素级精确**：直接从 Agg 渲染缓冲区取像素（`buffer_rgba`），
  与 `transData` 显示坐标一一对应，并内置自检（GT 曲线点必须落在墨迹附近）。

### 6.1 dataset-platform 适配器（Phase A.1）

`data/dataset_builder.py` 接入 materials-curve-dataset-platform（V0fix-final-2）
的生成器 API（参数采样 `sample_parameters` + 蠕变曲线模型
`generate_curve_data` + MCG-JSON 模式 + YOLO 标签收集），自行渲染并输出
baseline 同款侧车 + 平台生态格式：

```bash
# 训练数据（8 模板 × 概率采样，多曲线 1-5，含对数轴与退化）
python data/dataset_builder.py --out-dir data/train_platform --count 2000 \
    --seed 20260815 --yolo
# 单曲线可评估集（直接喂 scripts/evaluate.py）
python data/dataset_builder.py --out-dir data/eval_platform --count 100 \
    --num-curves 1 --seed 20260816
```

- 输出：PNG / 曲线掩码 / GT CSV（单曲线 `<stem>.csv`，多曲线
  `<stem>_cN.csv` + `<stem>_curves.json` 清单）/ `_meta.json` /
  `_labels.json` / `_mcg.json`（平台 MCG-JSON）/ `_yolo.txt`（可选）；
- 平台渲染器不实现模板 image_settings/key_effects、不支持对数轴、无退化、
  无刻度值 GT，且用 savefig 落盘 —— 适配器因此自渲染（buffer_rgba，
  像素级精确），平台仓库零改动；
- 修正/扩展：GBK 模板容错（平台自带 3 个 GBK 模板）、对数轴十年对齐
  （≥3 个数量级保证刻度充足）、图例 inside_lower_left 映射、YOLO bbox
  y 轴翻转、退化流水线。

## 7. 评估结果（stub OCR，mci 环境）

### 合成测试集（40 张，seed 20260806）

| 后端 | 中位 rel-RMSE | p90 | 最大 | ≤1% 达标率 | 失败数 |
|------|--------------|-----|------|-----------|--------|
| CV（训练免） | 1.73% | 89.2% | 82.4% | 40% | 0 |
| U-Net（基线，400 张合成训练） | 0.77% | 1.87% | 3.2% | 70% | 0 |
| **U-Net（Phase A，2400 张混合训练 @512）** | **0.40%** | 1.03% | 1.65% | **87.5%** | 0 |

### dataset-platform 测试集（100 张单曲线，seed 20260816，8 模板）

| 后端 | 中位 rel-RMSE | p90 | 最大 | ≤1% 达标率 | 失败数 |
|------|--------------|-----|------|-----------|--------|
| CV（训练免） | 1.07% | 77.9% | 5.41% | 49% | 0 |
| U-Net（基线，400 张合成训练） | 0.76% | 2.04% | 5.65% | 61% | 0 |
| **U-Net（Phase A，2400 张混合训练 @512）** | **0.20%** | 0.42% | 0.67% | **100%** | 0 |

评估指标：`rel_rmse = RMSE / y 满量程`（GT 在预测 x 处线性插值），
`x_coverage` = 预测 x 范围与 GT 重叠比例。Phase A 模型已达成验收目标
（中位 ≤0.5%、平台集 100% ≤1%）；合成集最大误差 1.65% 来自极端动态
范围曲线（线性 y 跨 4.6 个数量级的幂函数）。U-Net 单图推理（GPU）< 1 s，
全流程瓶颈在 CPU OCR（~6 s）。

### Phase A 关键改动（相对基线）

- 数据：`data/dataset_builder.py` 接入 dataset-platform（8 模板 × 概率采样、
  多曲线 1-5、线性/对数轴、退化），2400 张混合训练（400 合成 + 2000 平台）；
- 模型：512 分辨率 + AMP 混合精度 + 增强增强（随机裁剪/透视/扫描噪声），
  从 256 检查点续训（U-Net 全卷积，分辨率无关）；
- 管线：`_filter_mask_fragments`（U-Net 掩码骨架化前剔除标题/边框/尘点碎片，
  修复了追踪起点污染与列质心回退的灾难样本：19.1%→0.33%）；
  `configs/baseline.yaml` 默认 `segmenter: unet`、`unet_size: 512`；
- 修复：`DEFAULT_CONFIG_PATH` 目录层级错误（此前默认配置从未被读取，
  512 模型会被以 256 分辨率推理而精度退化）。

## 8. 多曲线 / 多子图扩展路线（已预留）

- **数据模型**：`ExtractionResult.curves` 已是 `List[Curve]`，`Curve` 带
  `color` 与 `legend_label` 字段；
- **接口**：`legend_matcher.py` 协议已定义（OCR 图例文本 + 颜色/空间关联），
  多曲线后端只需返回更多 `Curve` 并填充 label；
- **分割**：U-Net 输出改为 K 通道（每条曲线一通道）或实例分割后按组件分曲线；
- **检测**：`chart_structure` 换成 YOLOv8-nano（已在环境安装 ultralytics），
  输出同一 `ChartStructure` 数据类；
- **数据**：与 materials-curve-dataset-platform（V0fix-final-2）对接，
  在其 PNG+CSV+MCG-JSON 基础上增加掩码/YOLO 标签导出。

## 9. 已知限制（诚实的边界）

- CV 后端：浅色曲线（>150 灰度）、高密虚线、曲线贴底时刻度带污染、
  组件评分偶发选中边框/网格残留 → 用 `--segmenter unet` 规避；
- 坐标类型判别需要 ≥3 个有效刻度标签（合成数据已保证，真实论文图若只有
  2 个标签会在结果中给出 warning）；
- 默认假设：白底深色墨迹、左侧 y 轴、底部 x 轴；深底图（Otsu 兜底）与
  右轴/顶轴布局未覆盖；
- OCR 用 CPU paddle（Windows DLL 冲突），单图 ~6-15 s，<5 s 目标依赖
  Phase 2 的 GPU OCR 独立服务或更轻量识别模型。

## 10. 测试

```bash
python -m pytest tests -q
```

- `test_parse_number.py`：刻度文本解析（科学计数法/上标/Unicode 负号）；
- `test_coordinate.py`：线性/对数拟合、方向、不足刻度报错；
- `test_metrics.py`：RMSE/覆盖率指标正确性；
- `test_pipeline.py`：生成器自检 + CV 端到端（宽松断言）+ U-Net 端到端
  （严格断言：每图覆盖率 > 90%、rel_rmse < 1%，无模型时自动跳过）。
