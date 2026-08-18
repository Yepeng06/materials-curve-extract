# 下一步工作 Prompt — materials-curve-intel 项目（新对话交接版 v4 · 全局总结）

> AI 你好，这是项目阶段性交接文件。**你必须全文阅读后再开始工作。**
> 本文件可修改（不同于上级 Prompt.md）。
> 交接日期：2026-08-18（B-5a 完成 paddle 99%；Phase C 多曲线分割首轮完成召回 67.7%）
> **用户强调：无论何时都要先做深入研究（论文/社区文档/官方文档）再设计动手**。
> **真实文献曲线图仍在收集中**（Phase A.3 用户任务，到位后优先 B-5 真实图验证）。

## 〇.5、全局现状总结（2026-08-18 交接重点，先读这一节）

### 实现了什么（按时间线）
| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase A | 合成数据工厂、U-Net 512 曲线分割（val_iou 0.8166）、单曲线提取管线（结构→刻度→映射→分割→追踪→导出）、Web 演示 | ✅ |
| B-1 | 刻度 OCR 加固（条带放大/整图兜底/多尺度 tick 检测/RANSAC/端点锚定） | ✅ 刻度识别率 x 97.2%/y 96.3% |
| B-2 | 标题/轴标题/单位识别（title_reader）+ 竖排 y 标题 CW 旋转 OCR | ✅ 检出率 100% |
| B-3 | 三信号坐标判型（值序列+像素间距+外部先验） | ✅ 轴类型 x/y 96% |
| B-4 | YOLOv8-nano 结构检测（6 类，mAP50 0.942） | ✅ 可切换 structure_backend: cv|yolo |
| **B-5a** | OCR 刻度值误读增强（10^N 上标家族消歧/低分过滤/4x 重 OCR/标签中心像素/缺刻度补全） | ✅ **paddle 达标率 70%→99%** |
| **Phase C 首轮** | 多曲线实例分割（U-Net K=6 通道）、CSV 重建实例掩码、多曲线提取/评估管线 | 🟡 **召回 67.7%**（目标 95%） |

### 做得好（关键成功点）
1. **B-5a 三连突破**：候选解析+序列消歧（'100'=10⁰ 等）→ 86%；**标签中心像素**（CV mark 检测顶部漂移 +9.5px 是隐蔽根因）→ **99%**；pytest 106/106、stub 零回退
2. **Phase C 关键洞察链**：灰度聚类掩码不可靠（虚线/抗锯齿）→ curves_px 折线（16 稀疏点漂移）→ **CSV 重建掩码**（160 密集点经刻度标签映射，训练目标=评估基准，验证 0.4px）→ 召回 48%→67%
3. 严谨实验纪律：每步 pytest + 回归 + 诊断（12 张最差图逐刻度分析、失败分布量化、对照实验证明 512 是精度关键）

### 不够理想 / 问题在哪
| 问题 | 根因 | 现状 |
|------|------|------|
| Phase C 召回 67.7% vs 目标 95% | ① 87/175 条失败是 1-2% **边缘型**（模型精度边际，纯训练已平台化：512c 66.7%→512d 64.1% 无增益）；② 61 条 2-5%；③ 27 条 >5%（曲线靠近处**通道归属混淆**/追踪跳线） | 见"解决办法" |
| 单曲线召回 91.7%（5/60 失败） | 多任务 6 通道共享编码器精度略低于专用单曲线模型（单曲线模型 100%） | 可接受（单曲线场景可用单曲线模型） |
| 真实图验证未开始 | 用户真实论文图仍在收集 | 待用户 |
| 多曲线 Web 展示未接 | Phase C 未达标 | 后续 |

### 解决办法（候选路径，按性价比）
1. **更大模型**：UNet base 64→96（8GB 显存 batch 4，~2x 训练时间）——针对边缘型
2. **更多曲线形态训练数据**：扩展 dataset_builder 模板（陡峭尾部/交叉场景）
3. **图例颜色辅助归属**：真实图有图例时用颜色/标签关联（黑白合成图无效，真实图受益）
4. **推理侧**：阈值 0.3 + 跳变截断已落地（+1%）；细化半径 3 最优（69.2%）；语义细化失败（单曲线模型不支持多曲线语义）
5. **验收对齐**：与用户确认 95% 召回的口径（合成集 vs 真实图；含曲线数错误惩罚否）

### 后续计划（优先级）
1. **【需训练，先请示】Phase C 突破**：base 96 训练（~5h）或数据扩展，目标召回 ≥90%
2. **【依赖用户】B-5 真实图验证**：图到位后 --ocr paddle 评估 + gold 标注 RMSE
3. **【无需训练】5.2 多曲线评估基建**（legend_matcher 增强、evaluate 多曲线模式）——Phase C 验收需要
4. **【无需训练】5.4 Phase D 验收材料**（500 张独立 seed 验收集 + ablation 报告）
5. **【需训练，先请示】5.6 B-4 增强**（x_axis_line mAP 0.749 弱）或 OCR 微调

### 技术栈（现状）
- Python 3.11（Anaconda env **mci**）、PyTorch 2.13+cu126（GPU）、ultralytics 8.4.115（YOLOv8n）、
  PaddleOCR 3.7.0（CPU）、numpy 1.26.4（固定）、OpenCV/scikit-image/scipy、FastAPI + 原生前端
- 模型：U-Net 单曲线（unet_curve.pt 512，val_iou 0.8166）+ **多曲线 K=6（unet_multi_curve_512c.pt 512，val_iou 0.4993@CSV 目标）** + YOLOv8n 结构检测
- 管线：extractor 单一入口，segmenter: cv|unet|multi_unet；structure_backend: cv|yolo；ocr: stub|paddle

## 〇、先读这些（每次会话开始必读）

1. **`F:\CODE\New\Prompt.md`**（只读，禁止修改）— 项目唯一权威指导文件：竞赛目标（一等奖）、
   六大要求（前沿技术/效果第一/先请示/持怀疑态度调研/分工）、验收标准（RMSE ≤1% 满量程、
   坐标类型 ≥98%、单图 <5s GPU、多曲线召回 ≥95%）、技术路线（YOLOv8-nano + PaddleOCR + U-Net + 坐标映射）。
2. **`F:\CODE\New\baseline\README.md`** — baseline 完整架构/用法/评估/限制。
3. **`F:\CODE\New\baseline\AXIS_TEXT_RESEARCH.md`** — 坐标轴文字/数字鲁棒识别调研与设计（B-1~B-3 已实施）。
4. **`F:\CODE\New\baseline\B5A_DESIGN.md`** — B-5a 设计+实施记录（10^N 家族/标签中心像素等）。
5. **`F:\CODE\New\baseline\C_DESIGN.md`** — Phase C 多曲线分割设计+四轮训练记录+失败分析。
6. **`F:\CODE\New\baseline\TESTING_GUIDE.md`** — 用户实操测试指南。
7. **`F:\CODE\New\baseline\web\README.md`** — Web 演示系统说明。
8. 开始工作前：`cd F:\CODE\New\baseline && git log --oneline -30 && git status`。

## 一、项目背景与验收标准（摘要）

**主题：** 材料科学图像曲线智能识别与解析（自动从曲线图提取数据 → 结构化输出）。
**验收：** 曲线数据点 RMSE ≤ 坐标轴满量程 1%；坐标类型判断准确率 ≥98%；单张 <5s（GPU）；
多曲线召回 ≥95%（Phase C）。
**验收方式：** ① dataset-platform 独立 seed 500 张测试图；② ≥50 张真实论文蠕变图人工标注对比；
③ 每个模块 ablation。
**首批领域：** 蠕变曲线（做透后再泛化）。

## 二、环境与仓库状态

**环境：** Anaconda 虚拟环境 **`mci`**（Python 3.11，RTX 4060 8GB / CUDA 12.6）
- torch 2.13.0+cu126（GPU ✓）、ultralytics 8.4.115（YOLOv8n 检测已训练）
- paddlepaddle 3.3.1（**CPU 版**）+ paddleocr 3.7.0（真实 OCR ~10-30s/图含整图识别）
- numpy **必须固定 1.26.4**；fastapi 0.141.1 / uvicorn 0.52.3（Web 已装）

**仓库：** `F:\CODE\New\baseline`（git，master）
**归档：** `F:\CODE\New\baseline_backup_20260816_bphases`（B 系列完成后备份）
**数据工厂：** `F:\CLAUDE\NewProject1\materials-curve-dataset-platform`（V0fix-final-2，8 模板；平台仓库零改动）

**数据目录（均不入库，gitignore）：**
| 目录 | 内容 |
|------|------|
| `data/synthetic` | 40 张合成测试集（stub 基准 med ≤0.40%/达标 87.5%） |
| `data/train_platform` | 2000 张平台风格训练集（2-5 曲线，含掩码/CSV/侧车/MCG/YOLO 标签） |
| `data/train_platform_single` | 600 张单曲线训练集（B-5a/Phase C 补充） |
| `data/train_platform_4c` / `_5c` | 600 张 4 曲线 / 300 张 5 曲线训练集 |
| `data/eval_platform` | 100 张单曲线评估集（**paddle 99% 达标**） |
| `data/val_multi` / `val_single` | 120 张 4 曲线 + 60 张单曲线验证集（Phase C 评估） |
| `data/eval_*` | 历次评估输出（report.csv/summary.json，对比用） |
| `data/real_papers/` | 真实论文图收集区（README 已就位，**待用户收集**） |
| `data/failures/` | 真实失败样本库（fail_001 已恢复） |

**模型检查点（已入库或本地）：**
- `models/checkpoints/unet_curve.pt` — 单曲线 U-Net 512（val_iou 0.8166，**默认**）
- `models/checkpoints/unet_multi_curve_512c.pt` — **多曲线 K=6（当前最佳，召回 67.7%）**
- `models/checkpoints/unet_multi_curve_v3.pt` / `_512.pt` / `_512b.pt` / `_512d.pt` — 历史版本（v3=256、512/512b=折线目标、512d=CSV 目标继续训练无增益）
- `models/detection/yolo_struct.pt` — YOLOv8n 结构检测（mAP50 0.942）

## 三、当前进展（已完成）

### 3.1 提取管线（单曲线/单子图，接口稳定）
```
结构检测(CV 或 YOLO 可选) → 刻度OCR(PaddleOCR/Stub 双后端+整图兜底) → 坐标映射
(三信号判型+RANSAC+端点锚点+缺刻度补全) → 标题/轴标题/单位识别(B-2) → 曲线分割
(U-Net 512/CV) → 骨架追踪+亚像素细化 → CSV/JSON/overlay 导出
```

### 3.2 数据与精度（stub 路径基准，不得回退）
| 评估集 | med | 达标率 |
|--------|-----|--------|
| 合成 40 张 | 0.404% | 87.5% |
| 平台 100 张 | 0.204% | **100%** |

### 3.3 真实 OCR（paddle）路径精度（B-1→B-3→B-5a）
| 版本 | 达标率 | med | p90 | 轴类型 x/y |
|------|--------|-----|-----|-----------|
| 原始（B-1 前） | 24.5% | 0.43% | 39.7% | ~50%/50% |
| B-1 后 | 48% | 0.14% | 47.8% | 64%/79% |
| B-3 最终 | 70% | 0.0034% | 0.34% | 96%/96% |
| **B-5a 最终** | **99%** | **0.25%** | **0.58%** | **100%/100%** |
- B-5a 关键：10^N 上标误读家族消歧（'100'/'10'/'10-'/'012'/'0-2'/'102.'/'0-1' 等）、
  低分/非数字文本过滤、4x 条带重 OCR、**标签中心像素**（CV mark 顶部漂移 +9.5px）、
  缺刻度补全、strict 关联优化——详见 B5A_DESIGN.md
- 剩余 1 张（img_0060 1.6%）：亚像素级残余，待真实图阶段验证

### 3.4 Phase C 多曲线实例分割（首轮完成，召回 67.7%）
- 模型：UNet(K=6 通道) + MultiUNetSegmenter；训练 `train/train_segmentation_multi.py`
- **实例掩码 = GT CSV 重建**（160 密集点经 labels.json 刻度标签映射；与评估基准一致，
  验证 0.4px）——灰度聚类、curves_px 折线均已否决（原因见 C_DESIGN.md）
- 推理：`extract_curves_multi`（阈值 0.3 + 通道 argmax 互斥 + 尾部跳变截断）+
  extractor `--segmenter multi_unet`；评估 `scripts/eval_multi.py`
- **结果：召回 67.7%（单曲线 91.7%、多曲线 ~65%），曲线数准确率 92%**（178 张验证集）
- 失败分类：87 条 1-2% 边缘型 / 61 条 2-5% / 27 条 >5%（归属/追踪）
- 训练平台化：512c（CSV 目标）66.7% → 512d（+20 epoch）64.1% 无增益

### 3.5 测试与质量
- pytest **106/106**（+test_multi_segmentation.py 4 个）
- git 历史：A → B-1~B-4 → B-5a×2 → Phase C×3（见 git log）
- 性能：整图 OCR 共享（29.4s/图 paddle）；PaddleOCR 引擎类级缓存

## 四、待用户任务（Phase A.3，用户侧）
- **收集 ≥50 张真实论文蠕变图**（先 10-15 张）→ `data/real_papers/raw/` + gold 标注；
  测试中失败图放 `data/failures/` 或发路径

## 四.5、前期研究依据（新对话须复核扩充）
- 刻度文本/坐标校准：ChartEye（arXiv:2408.16123）、ICDAR CHART-Infographics、
  PlotQA 文本角色分类；US 专利 20190130614A1 序列校验；WebPlotDigitizer 自动刻度检测；
  PaddleOCR 多尺度/limit_side_len 陷阱（Discussion #14271）、垂直文本（3693449）
- 多曲线：**LineFormer**（arXiv:2305.01837 实例分割范式）、Socratic Chart
  （arXiv:2504.09764 掩码细化）、Efficient extraction of experimental data from
  line charts（Graphical Models 2025）、UW InfoVis 2018 颜色映射提取、
  thu-digitizer（确定性+校验工程化）
- 本阶段经验：训练目标必须与评估基准一致（CSV 重建掩码）；灰度聚类在虚线/抗锯齿下失效；
  512 分辨率是曲线位置精度关键；单曲线模型不支持多曲线语义分割

## 五、下一步计划（按优先级）

**执行纪律：** 每项开工前先做 ≥1 轮 web/论文调研复核，设计文档（1 页内）先给用户确认再实现。

### 5.1【已完成】B-5a：OCR 刻度值误读增强（70%→99%）

### 5.5【需训练，先请示】Phase C 突破（召回 67.7%→90%+）
- 候选①：UNet base 64→96（~2x 训练时间，8GB batch 4）——针对 87 条边缘型
- 候选②：dataset_builder 扩展曲线形态模板（陡峭尾部/交叉）——训练数据多样性
- 候选③：图例颜色辅助归属（真实图受益）
- 候选④：验收口径与用户对齐（真实图 vs 合成；曲线数错误处理）
- 训练命令（512c 继续）：
  `$PY train/train_segmentation_multi.py --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single --val-dir data/val_multi,data/val_single --epochs N --batch 8 --size 512 --per-dir-limit 600 --init models/checkpoints/unet_multi_curve_512c.pt --out models/checkpoints/unet_multi_curve_NEW.pt`

### 5.2【无需训练】多曲线评估基建完善（Phase C 验收需要）
- legend_matcher 增强（整图 OCR 图例文本 + 颜色映射）；evaluate.py 多曲线模式
- 多曲线指标：逐曲线 F1/RMSE/召回（eval_multi.py 已有基础版）

### 5.7【依赖用户】B-5 真实图验证
- 图到位后：`--ocr paddle` 评估（无 labels.json 不能走 stub）；失败图入 data/failures/ 回归

### 5.4【无需训练】Phase D 验收材料
- 500 张独立 seed 验收集（dataset_builder --num-curves 1 --count 500 --seed <新>）
- 测试报告整理（B-5a 99%、Phase C 67.7%、ablation 表）

### 5.3【无需训练】Web 演示增强（多曲线展示、失败图导出）

### 5.6【需训练，先请示】其他训练类
- B-4 增强：x_axis_line 弱（mAP50 0.749）→ 更大 imgsz/更长训练/标签细化
- OCR 专用模型（PaddleOCR 微调）——Phase 2 独立推理服务（GPU）

## 六、已知的技术坑（务必先读）

1. **Windows DLL 冲突**：torch 与 paddle 同进程互斥（WinError 127）。OCR 用 CPU paddle
   （enable_mkldnn=False）；PaddleOCRBackend._ensure 先 import torch 再 import paddle。
2. **numpy 固定 1.26.4**（mci 环境）。
3. **matplotlib 渲染差异**：GT 像素用 `fig.canvas.buffer_rgba()`，勿用 savefig 反推。
4. **conda run 偶发插件问题**——用 `F:\anaconda3\envs\mci\python.exe` 直接调用。
5. **paddle 评估慢**：~30s/图 → 100 张 ~50 分钟；评估期间勿并行 YOLO/U-Net 训练。
6. **'10N' 上标粘连**（'10²'→'102'）：resolve_values 序列一致性重解析（勿回退 B-5a 的候选消歧）。
7. **重复刻度**（'200'+'0' 双框）：<3px 去重 + B-5a 10px 双框去重（取高分/有值）。
8. **像素投票规则**：ratio ≥8 → log（次刻度密度），≤5 → linear，单一等距 abstain。
9. **竖排 y 轴标题**：matplotlib +90° → ROTATE_90_CLOCKWISE（CCW 读出碎片）。
10. **YOLO x_axis_line 弱**（mAP50 0.749）：plot_area 底边兜底。
11. **U-Net 推理分辨率必须等于训练分辨率**（512；`--unet-size N` 切换）。
12. **评价口径**：rel_rmse 以 GT y 满量程归一；GT 稀疏点不能直接比像素。
13. **gitignore 陷阱**：runs/ 与 *.pt 已 ignore；data/ 全量不入库。
14. **glob 过滤**：evaluate/run_baseline 必须过滤 `*_mask.png`。
15. **PaddleOCR 长条带**：超长条带被 resize 到 max_side_limit 4000 内变形——裁剪要合理。
16. **Phase C 教训**：① 灰度聚类掩码在虚线/抗锯齿下不可靠；② curves_px 是 16 稀疏点，
    不等于完整曲线（160+ 点 CSV）；③ 训练目标必须与评估基准一致（CSV 经 labels.json 映射）；
    ④ 512 是曲线位置精度关键（256 下 3.3px 偏差 → rel 1-2%）；⑤ 多曲线模型不支持单曲线语义
    分割（语义细化方案失败）；⑥ 纯训练收益会平台化（512c→512d 无增益），及时止损评估。
17. **512 训练显存**：batch 8 + 6 通道 OK（~8GB）；batch 16 可能 OOM。

## 七、工作约定（继承 Prompt.md 六大要求）

- 每次会话开始重读 `Prompt.md` + 本文件 + 相关设计文档 + git log/status；
- **（用户强调）无论何时先做深入研究再动手**：每项工作开工前 ≥1 轮 web_search/论文/社区/
  官方文档调研，形成 ≤1 页设计依据，先给用户确认设计再实现；
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

# 单曲线评估（stub=确定性快；paddle=真实 OCR 慢 ~50 分钟/100 张）
$PY scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval_x --ocr stub --segmenter unet
$PY scripts/evaluate.py --data-dir data/eval_platform --out-dir data/eval_x --ocr paddle --segmenter unet

# 多曲线训练（512 分辨率，~10min/epoch；--per-dir-limit 600 均衡子集）
$PY train/train_segmentation_multi.py --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single --val-dir data/val_multi,data/val_single --epochs 10 --batch 8 --size 512 --per-dir-limit 600 --init models/checkpoints/unet_multi_curve_512c.pt --out models/checkpoints/unet_multi_curve_NEW.pt

# 多曲线评估（召回/逐曲线 RMSE/曲线数；--size 512 必须匹配训练）
$PY scripts/eval_multi.py --data-dir data/val_multi,data/val_single --model models/checkpoints/unet_multi_curve_512c.pt --out-dir data/eval_multi_x --size 512

# 单图多曲线提取（multi_unet 模式）
$PY scripts/run_baseline.py --image data/val_multi/img_0000.png --out-dir data/outputs --ocr stub --segmenter multi_unet

# 数据生成
$PY data/dataset_builder.py --out-dir data/eval_xxx --count 100 --num-curves 1 --seed <新seed>
$PY data/dataset_builder.py --out-dir data/train_platform --count 2000 --seed 20260815 --yolo

# 测试
$PY -m pytest tests -q              # 当前 106/106
```

## 九、待用户确认/执行的事项

1. **真实论文图收集**（Phase A.3）：≥50 张蠕变图，先 10-15 张 → data/real_papers/raw/；
2. **Phase C 突破方向确认**：base 96 训练 / 数据模板扩展 / 图例辅助 / 验收口径（5.5）；
3. **验收口径**：95% 多曲线召回在什么数据集上考核（合成独立 seed？真实图？）；
4. 收集失败图（Web 上传测试失败）是回归样本的重要来源；
5. B-5a 剩余 1 张（img_0060）与 Phase C 剩余失败的复测材料已就绪。
