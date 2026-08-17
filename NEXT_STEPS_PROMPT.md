# 下一步工作 Prompt — materials-curve-intel 项目（新对话交接版 v3）

> AI 你好，这是项目阶段性交接文件。**你必须全文阅读后再开始工作。**
> 本文件可修改（不同于上级 Prompt.md）。
> 交接日期：2026-08-16（Phase A + B-1~B-4 全部完成；paddle 达标率 24.5%→70%）
> **下一对话优先做「不需要训练模型」的工作**（见 §五标注），训练类工作（Phase C 分割等）需先请示。
> **用户强调：无论何时都要先做深入研究（论文/社区文档/官方文档）再设计动手**——
> §五每项已附初步研究依据（2026-08-16 调研），新对话须先复核/扩充再实施。
> **真实文献曲线图仍在收集中**（Phase A.3 用户任务，到位后优先 B-5 真实图验证）。

## 〇、先读这些（每次会话开始必读）

1. **`F:\CODE\New\Prompt.md`**（只读，禁止修改）— 项目唯一权威指导文件：
   竞赛目标（一等奖）、六大要求（前沿技术/效果第一/先请示/持怀疑态度调研/分工）、
   验收标准（RMSE ≤1% 满量程、坐标类型准确率 ≥98%、单图 <5s GPU、多曲线召回 ≥95%）、
   技术路线（YOLOv8-nano + PaddleOCR + U-Net + 坐标映射引擎）。
2. **`F:\CODE\New\baseline\README.md`** — baseline 完整架构/用法/评估/限制说明。
3. **`F:\CODE\New\baseline\AXIS_TEXT_RESEARCH.md`** — 坐标轴文字/数字鲁棒识别调研
   与设计方案（B-1~B-3 已按其层 1/层 2/层 3 实施完成）。
4. **`F:\CODE\New\baseline\TESTING_GUIDE.md`** — 用户实操测试指南。
5. **`F:\CODE\New\baseline\web\README.md`** — Web 演示系统说明。
6. 开始工作前：`cd F:\CODE\New\baseline && git log --oneline -30 && git status`。

## 一、项目背景与验收标准（摘要）

**主题：** 材料科学图像曲线智能识别与解析（自动从曲线图提取数据 → 结构化输出）。
**验收：** 曲线数据点 RMSE ≤ 坐标轴满量程 1%；坐标类型（线性/对数）判断准确率 ≥98%；
单张处理 <5s（GPU）；多曲线召回 ≥95%（Phase C）。
**验收方式：** ① dataset-platform 独立 seed 500 张测试图；② ≥50 张真实论文蠕变图
人工标注对比；③ 每个模块 ablation。
**首批领域：** 蠕变曲线（做透后再泛化）。

## 二、环境与仓库状态

**环境：** Anaconda 虚拟环境 **`mci`**（Python 3.11，RTX 4060 8GB / CUDA 12.6）
- torch 2.13.0+cu126（GPU ✓）、ultralytics 8.4.115（YOLOv8n 检测已训练）
- paddlepaddle 3.3.1（**CPU 版**）+ paddleocr 3.7.0（真实 OCR ~10-30s/图含整图识别）
- numpy **必须固定 1.26.4**；fastapi 0.141.1 / uvicorn 0.52.3（Web 已装）
- 其余见 `requirements.txt`

**仓库：** `F:\CODE\New\baseline`（git，master，HEAD=交接提交）
**归档：** `F:\CODE\New\baseline_backup_20260816_bphases`（本轮 B 系列完成后完整备份，
含 .git；运行 robocopy 后台完成）
**数据工厂：** `F:\CLAUDE\NewProject1\materials-curve-dataset-platform`（V0fix-final-2，
8 模板；**平台仓库零改动**，全部扩展在 baseline 内）

**数据目录（均不入库，gitignore）：**
| 目录 | 内容 |
|------|------|
| `data/synthetic` | 40 张合成测试集（seed 20260806，含 GT 侧车） |
| `data/train_platform` | 2000 张平台风格训练集（8 模板×250、多曲线 1-5、log 轴、含掩码/CSV/侧车/MCG-JSON/YOLO 标签） |
| `data/eval_platform` | 100 张单曲线平台评估集（seed 20260816） |
| `data/eval_b2` | 30 张 B-2 评估集（seed 20260817，meta 含 title/轴标签 GT） |
| `data/detection/` | YOLO 训练数据布局（500 张副本 images+labels + dataset.yaml） |
| `data/failures/` | 真实失败样本库（fail_001 已恢复；继续收集） |
| `data/real_papers/` | 真实论文图收集区（README 指南已就位，待用户收集） |
| `data/eval_*` | 历次评估输出（report.csv/summary.json，对比用） |

**模型检查点：**
- `models/checkpoints/unet_curve.pt`（**默认** = 512 U-Net，val_iou 0.8166；已入库）
- `models/detection/yolo_struct.pt`（**YOLOv8n 结构检测 best**，mAP50 0.942；已入库；
  `yolo_struct_last.pt` 为 last）
- `runs/detect/runs/detect/mci_struct_full/`（YOLO 训练输出全量，本地未入库）

## 三、当前进展（已完成，2026-08-16）

### 3.1 提取管线（单曲线/单子图，接口稳定）
```
结构检测(CV 或 YOLO 可选) → 刻度OCR(PaddleOCR/Stub 双后端+整图兜底) → 坐标映射
(三信号判型+RANSAC+端点锚点) → 标题/轴标题/单位识别(B-2) → 曲线分割(U-Net 512/CV)
→ 骨架追踪+亚像素细化 → CSV/JSON/overlay 导出
```

### 3.2 数据与精度（stub 路径基准，不得回退）
| 评估集 | med | max | 达标率 |
|--------|-----|-----|--------|
| 合成 40 张 | 0.404% | 1.648% | 87.5% |
| 平台 100 张 | 0.204% | 0.672% | **100%** |
| 平台 100 张（YOLO 结构后端 ablation） | 0.241% | 0.720% | **100%** |

### 3.3 真实 OCR（paddle）路径精度（B-1~B-3 演进 + B-5a）
| 版本 | 达标率 | med | p90 | 轴类型 x/y |
|------|--------|-----|-----|-----------|
| 原始（B-1 前） | 24.5% | 0.43% | 39.7% | ~50%/50% |
| B-1 后 | 48% | 0.14% | 47.8% | 64%/79% |
| B-3 最终 | 70% | 0.0034% | 0.34% | 96%/96% |
| **B-5a 最终** | **86%** | **0.31%** | **1.25%** | **100%/100%** |
- 硬失败 2→0→0；刻度文本识别率 x 97.2% / y 96.3%（B-1 核心目标）
- B-5a 修复的误读家族：'100'(10⁰)、'10'(10⁰/10⁻¹)、'10-'/'012'/'0-2'/'.102'/'102.'/'0-1'、
  '101'-'103' 粘连、低分碎片、轴标题碎片混入、缺刻度（补刻）——详见 B5A_DESIGN.md
- 剩余 14% 失败（全部判型正确，1-4% 软失败）：log 轴低端对刻度像素系统偏移
  （~2.5px）的放大——信息层面不可自纠；候选：YOLO tick_label 框接入 / 真实图验证

### 3.4 B-2 标题/轴标题/单位识别（新能力）
- title_reader.py：区域+内容规则角色分类；variable/unit 提取；log/ln 先验→fit_axis
- 竖排 y 轴标题顺时针 90° 旋转 OCR（matplotlib +90°；CCW 实测读出碎片）
- 验证（30 张平台集）：title/x_label/y_label 检出率 100%，variable/unit 解析率 100%
- dataset_builder meta.json 新增 title/x_label/x_unit/y_label/y_unit GT 字段

### 3.5 B-3 坐标类型判别（三信号融合）
- axis_kind.py：值序列一致性（等差/等比 + '10N' 上标重解析 + 容错众数投票）
  + 像素间距（次刻度密度）+ 外部先验（B-2 log_hint），三票 ≥2 定类型
- 顺序：去重（<3px）→ 预 RANSAC（judge 前）→ 3 刻度单离群剔除 → judge → 拟合；
  R² 仅作 fallback 与质量分
- 2 刻度 10 的幂等比对 → log；模糊序列（100/101/102/103）重解析修复
- **轴类型准确率 x 96% / y 96%**（目标 ≥98%，差 2% 为 OCR 误读残留）

### 3.6 B-4 YOLOv8-nano 结构检测
- train/train_detection.py：ultralytics 数据集构建 + 训练 CLI（6 类：plot_area/
  x_axis_line/y_axis_line/tick_label/legend_box/axis_title）
- 全量训练（2000 张 80ep，yolov8n 640）：**mAP50 0.942**（x_axis_line 0.749 偏弱）
- detector.py：YOLO→ChartStructure（轴缺失时 plot 边缘兜底；tick_label→刻度位置；
  axis_title 框进 structure.meta）；extractor `structure_backend: cv|yolo`

### 3.7 测试与质量
- **pytest 89/89**；新增测试：test_tick_reader_b1.py（B-1）、test_coordinate.py 扩展
  （B-3）、test_title_reader.py（B-2）
- 性能：整图 OCR 结果在 title 识别与 legend 匹配间共享（37.8→29.4s/图）；
  PaddleOCR 引擎类级缓存（消除每图重建 ~20s）
- git 历史 30 提交（A → B-1 → B-2 → B-3×4 轮 → B-4 → 文档/性能）

## 四、待用户任务（Phase A.3，用户侧）
- **收集 ≥50 张真实论文蠕变图**（先 10-15 张）→ `data/real_papers/raw/` + gold 标注
  （指南已就位）；测试中遇到失败图直接放 `data/failures/` 或发路径

## 四.5、前期研究依据（2026-08-16 调研，新对话须复核扩充）

**刻度文本/坐标校准（5.1 相关）：**
- 学界范式：区域检测 + 文本角色分类（ChartEye arXiv:2408.16123；ICDAR CHART-Infographics
  IEEE 8978105；PlotQA）——B-2/B-4 已按此实现（YOLO axis_title + title_reader 角色分类）
- OCR 刻度值错误检测 = 序列一致性校验（US 专利 20190130614A1：robust to errors in the
  OCR ... wrong numerical value for the tick, an error detection）——与 B-3 的等差/等比
  校验思路一致（业界标准做法）
- 小字/低分辨率 OCR 增强：条带放大 2-4x、CLAHE、多尺度投票（PaddleOCR 官方优化实践：
  developer.baidu.com/article/detail.html?id=3696085、cloud.baidu.com/article/3693196；
  PaddleOCR Discussion #14271 limit_type/limit_side_len 参数陷阱）
- 坐标校准社区实践：WebPlotDigitizer DeepWiki 5.1 automated-detection-algorithms /
  4.2 data-extraction（自动刻度检测 + 颜色分离算法细节）

**多曲线提取（5.2/5.5 相关）：**
- **LineFormer**（arXiv:2305.01837）：折线图数据提取 = **实例分割**（而非语义分割），
  逐实例回归像素线——Phase C 实例分割的学术依据；社区已有 HuggingFace 权重
  （t29mato/lineformer-battery-finetuned，电池放电曲线微调版可参考）
- **Socratic Chart**（arXiv:2504.09764）：多代理协作图表理解 + 掩码形态学细化
  （腐蚀/膨胀/高斯模糊增强掩码）——掩码后处理可借鉴
- **Efficient extraction of experimental data from line charts**（Graphical Models
  2025, doi:10.1016/j.gmod.2025.101259）：端到端管线（找轴范围→逐线提取）
- **Extracting Color Mappings**（UW InfoVis 2018）：图例颜色映射提取——legend 匹配
  （图例色块 ↔ 曲线颜色）的算法依据
- 社区确定性方案：thu-digitizer（github.com/Rimagination/thu-digitizer）：
  evidence-first 数字化（确定性提取器 + 结果校验 + 可审计图库）——工程化思路可借鉴

## 五、下一步计划（按优先级；标注【无需训练】的优先做）

**执行纪律：** 每项开工前先做 ≥1 轮 web/论文调研复核上表依据，补充新文献后再设计；
设计文档（1 页内）先给用户确认再实现。

### 5.1【无需训练】B-5a：OCR 刻度值误读增强（✅ 完成 2026-08-16：70%→86%）
- 已实施：A 候选解析+序列消歧（axis_kind.py）；B 低分/非数字文本过滤（tick_min_score、
  _is_tick_label）；C 4x 条带重 OCR；D 缺刻度补全（fit_axis log gap-fill）；
  回归修复 3 轮（单调性/10N 平局偏好/3-tick 像素感知剔除/strict 关联/10px 去重）
- 剩余 14%：log 轴低端像素系统偏移放大（信息极限）——候选路径：
  ① YOLO tick_label 框接入（B-4 已训练）提供更精确刻度像素；② 真实图验证（B-5）；
  ③ CLAHE 未实施（评估中收益未知）；④ 竖排 y 刻度（平台数据无，真实图阶段做）

### 5.2【无需训练】Phase C 前半：多曲线评估基建
- 依据：LineFormer 实例分割思路（后段）；先做评估基建不依赖训练
- 方案（待调研复核）：① 多曲线指标（F1/召回、DTW、逐曲线 RMSE）——数据已就绪
  （train_platform 2000 张多曲线 + _curves.json + _cN.csv + 掩码）；② legend_matcher
  增强：整图 OCR 图例文本（已可读）+ 颜色映射关联（UW ColorMappings 思路）；
  ③ evaluate.py 多曲线模式（每曲线独立评估）

### 5.3【无需训练】Web 演示增强
- 结果卡片已展示标题/轴标题；可加：多曲线展示、失败图一键导出 data/failures/

### 5.4【无需训练】Phase D 准备（验收材料）
- 500 张独立 seed 验收集生成（dataset_builder --num-curves 1 --count 500 --seed <新>）
- 测试报告整理（全部评估表 + ablation：CV vs YOLO 结构、stub vs paddle、判型各信号）

### 5.5【需训练，先请示】Phase C 后半：多曲线分割
- **LineFormer 式实例分割**（arXiv:2305.01837）或 U-Net K 通道；数据已就绪
  （2000 张 2-4 曲线）；组件评分 top-K + 颜色分离；多曲线召回 ≥95% 验收

### 5.6【需训练，先请示】其他训练类
- B-4 增强：x_axis_line 检测弱（mAP50 0.749）→ 更大 imgsz/更长训练/标签细化
- OCR 专用模型（PaddleOCR 微调）——Phase 2 独立推理服务（GPU）

### 5.7【依赖用户】B-5：真实论文图验证
- 用户收集图到位后：`--ocr paddle` 评估（无 labels.json 不能走 stub）；失败图自动归
  入 data/failures/ 回归；gold 标注对比 RMSE


2. **numpy 固定 1.26.4**（mci 环境）。
3. **matplotlib 渲染差异**：GT 像素用 `fig.canvas.buffer_rgba()`，勿用 savefig 反推。
4. **conda 问题**：`conda run` 偶发插件报错/极慢（stderr 刷屏）——改用
   `F:\anaconda3\envs\mci\python.exe` 直接调用。
5. **paddle 评估慢**：整图 OCR（title 识别 + 兜底）每图 ~30s → 100 张 ~50 分钟；
   评估期间勿并行 YOLO 训练（CPU/GPU 争抢）。
6. **'10N' 上标粘连**：matplotlib log 标签 '10²' 被 OCR 读成 '102'（上标粘数字）；
   parse 层无法区分真 '100' → axis_kind.resolve_values 序列一致性重解析（勿回退）。
7. **重复刻度**：同一位置两个 OCR 框（'200'+'0'）→ fit_axis <3px 去重取高 score（勿回退）。
8. **像素投票规则**：单一等距间距 abstain（log/linear 主刻度都等距）；ratio ≥8 → log
   （次刻度密度），≤5 → linear；仅用有值刻度范围内的标记（超范围脏标记会误投 log）。
9. **竖排 y 轴标题**：matplotlib +90° → 旋转识别必须 **ROTATE_90_CLOCKWISE**
   （CCW 读出碎片）。
10. **YOLO x_axis_line 弱**（mAP50 0.749）：接入时 plot_area 底边兜底；不可依赖
    单靠 x_axis_line 检测。
11. **U-Net 推理分辨率必须等于训练分辨率**（512；`--unet-size N` 切换）。
12. **评价口径**：rel_rmse 以 GT y 满量程归一；GT 稀疏点不能直接比像素。
13. **gitignore 陷阱**：父目录排除后子文件无法 re-include（data/* + !data/*.py；
    !web/examples/*.png 须指向文件）；runs/ 与中间检查点不入库；data/ 全量不入库。
14. **glob 过滤**：evaluate/run_baseline 必须过滤 `*_mask.png`。
15. **PaddleOCR 长条带**：超长条带被 resize 到 max_side_limit 4000 内变形——条带
    裁剪要合理。

## 六、已知的技术坑（务必先读）

1. **Windows DLL 冲突**：torch 与 paddle 同进程互斥（WinError 127）。OCR 用 CPU paddle
   （enable_mkldnn=False）；PaddleOCRBackend._ensure 先 import torch 再 import paddle
   （类级缓存 + 初始化锁，勿回退）。

## 七、工作约定（继承 Prompt.md 六大要求）

- 每次会话开始重读 `Prompt.md` + 本文件 + `AXIS_TEXT_RESEARCH.md` + git log/status；
- **（用户强调，2026-08-16）无论何时先做深入研究再动手**：每项工作开工前 ≥1 轮
  web_search/论文/社区/官方文档调研（优先 arXiv、IEEE、官方仓库、DeepWiki），形成
  1 页内设计依据（§四.5 已有基础），先给用户确认设计再实现；实现中遇技术决策同样
  先查证再定；
- 一切工作先调研（论文/官方文档/社区）再设计，**先请示用户再执行**；
- 每步改动跑 `python -m pytest tests -q` 与 `scripts/evaluate.py --ocr stub` 回归，
  **指标不得回退**（stub 基准：合成 med ≤0.40%/87.5%、平台 ≤1% 达标率 100%）；
- 保持「模块接口稳定、逐步替换实现」；`data/` 不入库（代码例外），模型与训练数据本地；
- 定期归档：重大里程碑后复制 `baseline` 为 `baseline_backup_<日期>`（含 .git）。

## 八、常用命令速查（工作目录 F:\CODE\New\baseline）

```bash
# 注意：优先用 F:\anaconda3\envs\mci\python.exe（conda run 偶发插件问题）
PY=F:\anaconda3\envs\mci\python.exe

# Web 演示
$PY web/app.py                      # http://127.0.0.1:8000

# 评估（stub=确定性快；paddle=真实 OCR 慢 ~50 分钟/100 张）
$PY scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval_x --ocr stub --segmenter unet
$PY scripts/evaluate.py --data-dir data/eval_platform --out-dir data/eval_x --ocr paddle --segmenter unet
$PY scripts/evaluate.py --data-dir data/eval_platform --out-dir data/eval_x --ocr stub --segmenter unet --structure-backend yolo
$PY scripts/eval_axis_stats.py --report data/eval_x/report.csv   # 刻度识别率/轴类型准确率
$PY scripts/eval_title_stats.py --report data/eval_b2_out/report.csv --data-dir data/eval_b2  # B-2 标题解析率

# 数据生成
$PY data/dataset_builder.py --out-dir data/eval_xxx --count 100 --num-curves 1 --seed <新seed>
$PY data/dataset_builder.py --out-dir data/train_platform --count 2000 --seed 20260815 --yolo

# YOLO 训练
$PY train/train_detection.py --data-dir data/train_platform --out data/detection --epochs 80 --batch 16 --imgsz 640 --device 0 --name mci_struct_x

# 单图 + 调试（--debug 输出结构图/OCR 框/掩码图）
$PY scripts/run_baseline.py --image data/failures/fail_001_axis_detection.png --out-dir data/outputs --ocr paddle --segmenter unet --debug

# 测试
$PY -m pytest tests -q              # 当前 89/89
```

## 九、待用户确认/执行的事项

1. **真实论文图收集**（Phase A.3）：≥50 张蠕变图，先 10-15 张 → data/real_papers/raw/；
2. **Phase C 多曲线分割（需训练）**启动前请示（先做 5.1/5.2 无需训练部分）；
3. 收集失败图（如 Web 上传测试失败）是回归样本的重要来源；
4. 本交接 v3 已标注【无需训练】优先项，下一对话可直接开工。
