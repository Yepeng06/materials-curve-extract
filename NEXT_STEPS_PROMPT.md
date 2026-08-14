# 下一步工作 Prompt — materials-curve-intel 项目（继续开发用）

> AI 你好，这是项目的阶段性交接文件。你必须全文阅读后再开始工作。
> 本文件是开发指导文件，可修改（不同于上级 Prompt.md）。

## 〇、先读这些（每次会话开始必读）

1. **`F:\CODE\New\Prompt.md`**（只读，禁止修改）— 项目唯一权威指导文件：
   竞赛目标（一等奖）、六大要求（前沿技术/效果第一/先请示/持怀疑态度调研/
   分工）、验收标准（RMSE ≤1% 满量程、坐标类型准确率 ≥98%、单图 <5s GPU、
   多曲线召回 ≥95%）、技术路线（YOLOv8-nano + PaddleOCR + U-Net + 坐标映射引擎）。
2. **`F:\CODE\New\baseline\README.md`** — baseline 的完整架构/用法/评估/限制说明。

## 一、当前状态（已完成的 Baseline + Phase A.1 进行中）

**位置：** `F:\CODE\New\baseline`（已归档到 `F:\CODE\New\baseline_backup_20260814`，含
git 历史、400 张训练数据、模型 checkpoint；`.gitignore` 排除了 `data/`，未入库）

**环境：** Anaconda 虚拟环境 **`mci`**（Python 3.11，RTX 4060 / CUDA 12.6 已验证）
- torch 2.13.0+cu126（GPU ✓）、ultralytics 8.4.115（已装，Phase 2 用）
- paddlepaddle 3.3.1（**CPU 版**）+ paddleocr 3.7.0（CPU 推理 ~6s/图）
- numpy **必须固定 1.26.4**；其余见 `requirements.txt`

**已实现（单曲线、单子图）：**
```
结构检测(CV) → 刻度OCR(PaddleOCR/Stub 双后端) → 坐标映射(线性/对数自动判别)
→ 曲线分割(U-Net可训练 / CV后备，--segmenter 切换) → 骨架追踪+概率图亚像素细化
→ 像素→数据映射 → CSV/JSON/叠加图导出
```

**关键指标（40 张合成测试集，stub OCR，seed 20260806）：**

| 后端 | 中位 rel-RMSE | 最大 | ≤1% 达标率 | 每图点数 |
|------|--------------|------|-----------|---------|
| CV（训练免） | 1.73% | 82%（灾难样本存在） | 40% | ~800 |
| **U-Net（推荐）** | **0.77%** | 3.2% | 70% | 788（245–1612） |

- 单元+端到端测试 **59/59 通过**（`python -m pytest tests -q`）
- U-Net 训练：`train/train_segmentation.py`（400 张合成图+掩码，40 epochs ≈ 16 分钟，
  val IoU 0.78 / Dice 0.87，checkpoint 在 `models/checkpoints/unet_curve.pt`）
- U-Net GPU 推理 ~0.1s/图；真实 PaddleOCR 路径端到端已验证
- 合成数据工厂 `scripts/gen_synthetic.py`：5 类曲线 × 线性/对数轴 × 论文/实验风格 ×
  5 种退化，GT 像素级精确（Agg `buffer_rgba` + 自检），输出 PNG/CSV/掩码/meta/labels

**多曲线/多子图扩展预留：** `Curve` 已是列表、`legend_matcher.py` 协议就位、
`ChartStructure` 数据类是 YOLO 检测器的替换点、U-Net 可扩为多通道。

### Phase A.1 进展（2026-08-14 会话）

**`data/dataset_builder.py` 已完成并入库**（.gitignore 已改为 `data/*` + `!data/*.py`，
注意 git 的父目录排除规则陷阱）：

- 接入 `F:\CLAUDE\NewProject1\materials-curve-dataset-platform`（V0fix-final-2）生成器
  API（`sample_parameters` / `generate_curve_data` / MCG-JSON / quality_check /
  collect_yolo_labels），**平台仓库零改动**；自行用 `buffer_rgba` 渲染保证 GT 像素精确
  （平台渲染器不实现模板 settings、无对数轴、无退化、无刻度值 GT、savefig 落盘）
- 扩展：对数轴（十年对齐 ≥3 数量级）、退化流水线（含 low_quality 截图风）、模板
  image_settings/axis_settings 生效、GBK 模板容错（平台自带 3 个 GBK 模板，其
  generate_training_data.py 的 utf-8 加载会崩）、inside_lower_left 图例修正、
  YOLO bbox y 轴翻转修正
- 输出：PNG / 掩码 / CSV（单曲线）或 `_cN.csv`+`_curves.json`（多曲线）/ meta /
  labels（stub OCR）/ `_mcg.json`（平台格式）/ `_yolo.txt`（--yolo）
- 用法：`python data/dataset_builder.py --out-dir data/train_platform --count 2000
  --seed 20260815 --yolo`；`--num-curves 1` 生成可评估单曲线集
- **已生成**：`data/train_platform`（2000 张，8 模板×250，log 轴 1096 张，多曲线
  1-5 张混合，0 失败）+ `data/eval_platform`（100 张单曲线评估集，seed 20260816）

**U-Net 重训 Stage 1（256，混合数据 2400 张，40 epochs，AMP）：**
`models/checkpoints/unet_curve_256_v1.pt`，评估对比（stub OCR）：

| 评估集 | 旧模型（400 张合成训练） | 新模型 v1（2400 张混合训练） |
|--------|------------------------|------------------------------|
| 合成 40 张 | med 0.77% / max 3.18% / 70% | med 0.66% / max 2.82% / 65% |
| 平台 100 张 | med 0.76% / max **5.65%** / 61% | med **0.61%** / max **1.90%** / **88%** |

→ 混合数据显著消除灾难样本（平台集 max 5.65%→1.90%，达标率 61%→88%）；
合成集达标率略降 5pt（2 张恰好越过 1% 线，整体中位改善）。**续训中**：
`unet_curve_256_v2.pt`（--init 续 40 epochs）；之后按需 512 实验
（`--size 512`，全卷积架构可直接 `--init` 256 检查点；batch 4 + AMP，
~0.43s/batch）。评估 512 模型时需 `scripts/evaluate.py --unet-size 512` 或
config 覆盖 `unet_size`。

**训练/评估脚本新增能力**（均已入库）：`--size`、`--init`、`--limit`、
`--amp`（CUDA 自动开）、`--data-dir` 逗号分隔多目录、`evaluate.py --unet-size`、
`Extractor(config_override=...)`。

**Phase A.3 真实图收集**：`data/real_papers/README.md` 已写好收集+gold 标注指南
（用户任务，先 10-15 张即可开始）。

## 二、下一步工作（按优先级，每步先调研后动手、先请示用户再执行）

### Phase A：数据与精度夯实（建议先做，直接提升验收指标）
1. **接入 `F:\CLAUDE\NewProject1\materials-curve-dataset-platform`（V0fix-final-2）**
   - 调研其 generator API 与输出格式（PNG+CSV+MCG-JSON），写适配器
     `data/dataset_builder.py`：把平台输出转成本 baseline 的掩码/标签格式
   - 目标：训练数据 400 → 2000+ 张，覆盖更多曲线形态与模板
2. **U-Net 精度提升**（当前中位 0.77%，最大 3.2%，距"全部 ≤1%"有差距）
   - 512 分辨率训练（当前 256，需同步改 `train/train_segmentation.py` 的 SIZE 与
     `configs/baseline.yaml` 的 unet_size）、更长训练（60-80 epochs）、
     数据增强加强（随机裁剪/透视/模拟扫描噪声）
   - 重训后跑 `scripts/evaluate.py --segmenter unet` 对比，目标：中位 ≤0.5%、
     ≤1% 达标率 ≥90%
3. **真实论文图验证**（验收方式 2：≥50 张蠕变曲线论文图）
   - 用户搜集 ≥50 张真实蠕变图（可先 10-15 张），建立 `data/real_papers/`
   - 真实图需要 PaddleOCR 路径（`--ocr paddle`），注意 CPU OCR 慢；
     手工标注 gold set（用户配合 10-15 张），评估真实图精度

### Phase B：坐标与结构升级（对应技术路线）
4. **坐标类型判别加强**：当前靠 ≥3 刻度拟合残差（R² 判据）。调研后补充
   「刻度间距分析 + 轴标题文本先验（OCR 读 "log Time" 等）」；目标准确率 ≥98%
5. **YOLOv8-nano 结构检测**（替换 `chart_structure.py` 的 CV 检测）
   - 用平台/合成数据生成 YOLO 标签（plot_area/轴线/刻度/图例框/曲线框）
   - `models/detection/` + `train/train_detection.py`（ultralytics 已装）
   - 输出同一 `ChartStructure` 数据类，管线其余部分不动

### Phase C：多曲线提取（Prompt 验收项：多曲线召回 ≥95%）
6. **多曲线分割**：U-Net 输出 K 通道（每条曲线一通道）或实例分割；组件评分逻辑
   改为 top-K + 颜色分离
7. **图例匹配**：实现 `legend_matcher.py`（OCR 图例文本 + 曲线颜色/线型特征关联），
   填充 `Curve.legend_label`
8. **多曲线评估指标**：扩展 `eval/metrics.py`（多曲线 F1/召回，DTW）

### Phase D：工程化与成果（竞赛交付）
9. **Web 演示系统**：`web/app.py`（FastAPI + 轻量 HTML/JS，上传→结果+下载），
   复用 `Extractor`
10. **测试报告/软著/专利/论文**：按 Prompt.md 第九、十节分工执行

## 三、已知的技术坑（务必先读，避免重复踩坑）

1. **Windows DLL 冲突**：torch 与 paddlepaddle-gpu 的 cuDNN 同进程互斥
   （WinError 127，双向）。已定方案：OCR 用 CPU paddle（`enable_mkldnn=False`
   必须设，否则 PIR 崩溃）；paddleocr 导入链会加载 torch，`tick_reader.py`
   的 `_ensure` 里**必须先 import torch 再 import paddle**。
2. **numpy 版本**：mci 环境固定 **numpy==1.26.4**（paddle 3.3 与 numpy 2.x 不兼容）。
3. **matplotlib 渲染差异**：不同 matplotlib 版本（3.10 vs 3.11）渲染的刻度位置
   有差异，**数据生成与评估必须用同一环境（mci）**；GT 像素映射用
   `fig.canvas.buffer_rgba()`（与 transData 坐标严格一致），**不要用 savefig 反推**。
4. **Otsu 阈值失效**：浅色曲线（>150 灰度）会被 Otsu 判为背景；
   二值化用 `utils.ink_mask`（自适应阈值 + 深底图 Otsu 兜底）。
5. **刻度检测带污染**：x 刻度带必须限制在 y 轴右侧（y 轴标签字形会混入）；
   刻度列若与绘图区内部墨迹连通（曲线贴底）则拒绝；刻度-标签关联容差 25px，
   stub 标签（anchored=True）直接用标签中心像素（精确）。
6. **网格线去除**：网格必过刻度 → 「行/列覆盖率>45% 且与刻度 ±1px 对齐」删除
   （虚线免疫）；Hough 线段按截距聚类防误删曲线直线段；删除后 (7,3)/(3,7)
   闭运算桥接曲线缺口（大核会熔成大团，已踩过）。
7. **虚线曲线**：骨架端点「方向对齐+异组件+端点仅用一次」桥接（≤48px），
   必须在组件选择之前做；U-Net 训练掩码按实线绘制让网络学补全虚线。
8. **U-Net 推理分辨率必须等于训练分辨率**（256），否则精度下降。
9. **评价口径**：`rel_rmse` 以 GT y 满量程归一；GT 稀疏采样点不能直接比像素
   （陡峭段错位），用稠密链最近邻/插值。
10. **evaluate.py / run_baseline.py 的 glob 要过滤 `*_mask.png`**（已修，勿回退）。

## 四、工作约定（继承 Prompt.md 六大要求）

- 每次会话开始重读 `Prompt.md` 与本文档；新对话继续开发前先 `git log` 查看
   `baseline` 仓库历史，`git status` 确认工作区状态
- 一切工作先调研（论文/官方文档/社区）再设计，先请示用户再执行
- 每步改动跑 `python -m pytest tests -q`（mci 环境）与
  `scripts/evaluate.py --ocr stub --segmenter unet` 回归，指标不得回退
- 保持「模块接口稳定、逐步替换实现」的架构原则；`data/` 不入库，模型
  checkpoint 与训练数据在本地
- 定期归档：重大里程碑后复制 `baseline` 为 `baseline_backup_<日期>`（含 .git）

## 五、常用命令速查（conda activate mci 后，工作目录 F:\CODE\New\baseline）

```bash
python scripts/gen_synthetic.py --out-dir data/synthetic --count 40 --seed 20260806
python scripts/run_baseline.py --image data/synthetic/img_0001.png \
    --out-dir data/outputs --ocr stub --segmenter unet --debug
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval \
    --ocr stub --segmenter unet
python train/train_segmentation.py --data-dir data/train_synthetic \
    --val-dir data/synthetic --epochs 40 --batch 16 --out models/checkpoints/unet_curve.pt
python -m pytest tests -q
```

> 交接日期：2026-08-14
