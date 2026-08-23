# 下一步工作 Prompt — materials-curve-intel 项目（新对话交接版 v5 · 全局总结）

> AI 你好，这是项目阶段性交接文件。**你必须全文阅读后再开始工作。**
> 本文件可修改（不同于上级 Prompt.md）。
> 交接日期：2026-08-19（**Phase C 根因突破 + 修复重训暂停在 epoch 4**）
> 更新：2026-08-20（**v6：重训完成 val_iou 0.7636 + 方案 C Phase 1 实施 + Phase D 验收**）
> 更新：2026-08-20（**v7：云端 AutoDL 实验轮——增强 flip bug 修复、768/512-fixed 对照、三模型结论；实例已关机**）
> **用户强调：无论何时都要先做深入研究（论文/社区文档/官方文档）再设计动手**，
> 研究充分后选择最合适的方法，方案/计划必须经过充分调研。
> **真实文献曲线图仍在收集中**（Phase A.3 用户任务，到位后优先 B-5 真实图验证）。

## 〇.5、全局现状总结（2026-08-19 交接重点，先读这一节）

### 实现了什么（按时间线）
| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase A | 合成数据工厂、U-Net 512 曲线分割（val_iou 0.8166）、单曲线提取管线（结构→刻度→映射→分割→追踪→导出）、Web 演示 | ✅ |
| B-1~B-4 | 刻度 OCR 加固、标题/轴标题/单位识别、三信号坐标判型、YOLOv8n 结构检测 | ✅（详见历史） |
| B-5a | OCR 刻度值误读增强（10^N 上标消歧/标签中心像素等） | ✅ **paddle 达标率 70%→99%** |
| Phase C 首轮 | 多曲线实例分割（U-Net K=6）、CSV 重建掩码、多曲线提取/评估管线 | 🟡 召回 67.7%（目标 95%） |
| **C-2 根因修复（本会话）** | **发现并修复「训练目标 ≠ 评估基准」根因**：训练侧 _fit_axis_from_labels 底部标签过滤（c[1]<img_h-85）排除最底部 y 刻度 → **y 轴 25.5%（904/3548）+ x 轴 1.8%（65/3680）训练图把 log 轴误判 linear**，log 轴上训练掩码错位 100+px。修复：训练改走与评估完全相同的代码路径（detect_structure→read_ticks→build_axes→value_to_pixel）重建掩码 | ✅ 已修复并验证（240/240 通道对齐；pytest 110/110） |
| **C-2 重训验证** | 修复目标 + EMA(0.999) + 骨架召回损失（clDice 族），25 epoch 从 512c 续训 | ✅ **完成：evalfix_r2 val_iou 0.7636（+53% vs 512c）**；后处理升级（independent+min_area450）后 val 集 6a/6b=0.7185/0.7004，Phase D 多曲线 6a 0.7869、单曲线 99.4% ≤1% |
| **2026-08-20 云端实验轮** | AutoDL 4090：增强 flip bug 修复（cv2 3D 陷阱）、768 分辨率实验、512-fixed 对照 | ✅ 完成（详见 3.7/10.3.5）；实例已关机 |
| **Phase D 材料** | 500 张多曲线 + 500 张单曲线验收集（独立 seed 20260819） | ✅ 已生成（data/eval_phased_500 / _single） |

### 做得好（关键成功点）
1. **B-5a 三连突破**：候选解析+序列消歧 → 86%；**标签中心像素**（CV mark 顶部漂移 +9.5px 隐蔽根因）→ **99%**；pytest 全绿、stub 零回退
2. **Phase C 关键洞察链**：灰度聚类不可靠 → curves_px 折线漂移 → **CSV 重建掩码**（训练目标=评估基准）→ 48%→67%；**本会话再进一步：连轴映射代码路径都必须与评估一致**（25.5% log 轴图因标签过滤误判 linear，掩码错位 100+px）
3. **根因实证闭环**：全数据核查（904/3548 不一致）→ 像素级诊断（模型概率峰值 +1~2px、log 轴 1px≈2% rel）→ 修复后验证（240/240 通道对齐、epoch 1 val_iou 0.676→epoch 4 0.7345 远超旧模型）
4. **三路并行深度调研**（LineFormer 6a/6b、ChartZero GOI、clDice/SRL、EpiCurveBench、Graph-FINDER、STU-Net、Focal Tversky 等 25+ 来源），方案按「证据强度 × 成本」矩阵决策
5. 严谨实验纪律：每步 pytest + 回归 + 诊断（本会话新增 5 个诊断脚本：逐曲线失败分类/错误模式/像素偏差/概率峰值/映射一致性）

### 不够理想 / 问题在哪
| 问题 | 根因 | 现状 |
|------|------|------|
| ~~重训未跑完~~ | 已续训完成（evalfix_r2 0.7636） | ✅ 另发现并修复 **cv2.flip 3D 增强 bug**（历史训练 50% 样本目标错位，详见坑 17） |
| Phase C 召回 67.7%（修复前基线） | **根因已定位**：训练目标与评估基准不一致（见上）；叠加失败分布：1-2% 边缘 83 条（log 轴 1px≈2% rel + 概率峰值 +1~2px）、2-5% 47 条、>5% local_spike 41 条（尾部集中 decile 8-9 占 68%，**曲线交叉/靠近处 argmax 互斥丢弃通道像素**——实验证实交叉重叠区占图内墨迹 10-32%）、虚检 19 图（pred=5） | 修复已落地，重训验证中；local_spike 待新模型复测后决定 Hungarian 分支配对 |
| 6b 口径虚检惩罚 | 过分割/幽灵通道（面积 51-625px 碎片，部分与真曲线同量级） | min_area 过滤已实现（曲线数准确率 47/60→57/60，6b +1.5pp，6a -0.8pp 需谨慎调参） |
| 单曲线召回 91.7%（5/60 失败） | 4/5 失败图是 log 轴，且训练侧旧 fit 误判（img_0036 y=fit None→掩码全零；img_0037/0051 y=linear vs log）——**属同一根因** | 修复后预期单曲线也回升，待重训验证 |
| 真实图验证未开始 | 用户真实论文图仍在收集 | 待用户（图到位后 --ocr paddle 评估） |

### 解决办法（候选路径，按证据×成本，2026-08-19 调研后排序）
| 方案 | 证据 | 成本 | 状态 |
|------|------|------|------|
| A. 训练目标修复（统一映射路径） | 根因实证（904 例不一致） | 已实施 | ✅ 完成（evalfix_r2 0.7636） |
| B. EMA + 骨架召回损失（clDice 族） | clDice/SRL（ECCV24）细长结构强证据 | 低（已实施） | ✅ 已随重训 |
| C. 交叉分支配对 + Hungarian（纯后处理） | 对应 local_spike 41 条；LineFormer/追踪文献 | 中 | 📋 Phase 1（independent 多标签）已实施（73fe133）并显著提升 6a（0.607→0.722）；Phase 2 分支配对暂缓（决策门槛见 C_PAIRING_DESIGN.md） |
| D. 虚检抑制（min_area） | LineFormer 社区实证（过检 +62.3%→微调 -52.7pct） | 低（已实现） | ✅ **已默认开（multi_min_area: 450）**：6b 0.7004 历史最高，曲线数 168/180 |
| E. GOI 损失（嵌入头+正交+merge） | ChartZero 消融（IoU 0.75→0.82） | 高（改架构） | 📋 观察 A+B 结果后决定 |
| F. CoordConv / GroupNorm | ChartZero 消融（+0.06 IoU） | 低 | 📋 随 E 或独立验证 |
| G. 6a/6b 双口径 + 薄线子集指标 | LineFormer/EpiCurveBench | 低（已实现 6a/6b） | ✅ eval_multi_6ab.py |
| H. 更大模型 base 64→96/128 / 分辨率 768-1024 | STU-Net（2.1M→8.4M 真实收益）；细线任务分辨率首要杠杆 | 高（本地 8GB 受限；可租 GPU ~¥2-10/次） | 📋 **768 分辨率已实测：无端到端收益（6a -1.3pp，后处理像素阈值尺度敏感）**；base 96/128 未测（待 --base 参数化，需先请示） |
| I. 图例颜色辅助归属 / VLM 语义先验 | WebPlotDigitizer/PlotPick（VLM 召回 88-96%） | 中 | 📋 真实图阶段优先 |
| J. 曲线形态模板扩展（陡峭尾部/交叉） | 合成多样性>数量（域随机化经典证据） | 中 | 📋 若重训后仍不足 |

### 2026-08-23 状态（方案 E GOI 完成）
- **GOI 完整训练（云端 25ep）**：val_iou **0.7740** 历史最高；val 集 6a/6b = **0.7352/0.7192**（+embed_merge）；**Phase D 500 图 6a=0.8065（+2.0pp）/ 6b=0.7306（+1.7pp）**
- **默认配置已切**：multi_unet_checkpoint=unet_multi_goi.pt + multi_embed_merge: true；pytest 118/118
- 剩余差距：1-2% 边缘桶 69（候选方案 K 亚像素）、>5% 桶 31（primary_obvious 20 条；Phase 2 分支配对）
- 云端实例（新 4090）已完成本轮任务，**待关机**

### 2026-08-21 状态（本轮收尾）
- **方案 E（GOI）已实现**（commit 3f08b3e，pytest 118/118）+ 本地 5-epoch 预训练验证稳定（曲线数 173/180 历史最高）；云端 25-epoch 完整训练待 4090 空闲（开机排队中）
- **候选方案 K（亚像素细化）设计注记**：C_SUBPIXEL_DESIGN.md（K1 偏差校正→K2 质心修正→K3 Steger），针对 1-2% 边缘桶 67 条
- **Ablation 矩阵状态已更新**（PHASE_D_ACCEPTANCE.md：A5/A6/A7/A8 已做，A9 进行中）
- 云端实例：AutoDL 027 机 4090；GOI 训练完成后需提醒用户关机

### 后续计划（优先级）
1. **【当前最佳】evalfix_r2 + independent + min_area450**：val 6a/6b=0.7185/0.7004；Phase D 多曲线 6a 0.7869、单曲线 99.4%。距 95% 目标差距 = 交叉归属（local_spike）+ 边缘精度
2. **【待用户决策】下一轮实验**（云端已就绪，实例关机）：E（GOI 损失，需实现+确认）/ H-base（base 96，需 --base 参数化，~15min）/ 或真实图验证（依赖用户图）
3. **【评估基建】Phase D 复核**：可把 768/512fix 模型也跑 500 图验收（云端 ~1-2h）
4. **【依赖用户】B-5 真实图验证**：图到位后 --ocr paddle 评估 + gold 标注 RMSE
5. **【可选】Web 多曲线展示**（multi_unet 已接入，默认模型已切 evalfix_r2）
6. **云端注意**：AutoDL 027 机 ¥1.98/h 按量，无卡模式免费；凭据/脚本见第十节

### 技术栈（现状）
- Python 3.11（Anaconda env **mci**）、PyTorch 2.13+cu126（GPU）、ultralytics 8.4.115、PaddleOCR 3.7.0（CPU）、numpy 1.26.4（固定）、OpenCV/scikit-image/scipy、FastAPI + 原生前端
- 模型：U-Net 单曲线（unet_curve.pt 512，val_iou 0.8166）+ **多曲线 K=6**：**unet_multi_curve_evalfix_r2.pt（val_iou 0.7636，6a/6b 最佳组合，默认）**、unet_multi_512fix.pt（val_iou 0.7714，fixed-aug，6a 0.7093）、unet_multi_768.pt（val_iou 0.7531，曲线数 172/180 最高）、evalfix.pt（epoch 4）、512c（0.4993）+ YOLOv8n 结构检测
- **推理配置（2026-08-20 起）**：multi_independent_mask: true（方案 C Phase 1 多标签）+ multi_min_area: 450（虚检抑制）——val 集 6a/6b = 0.7185/0.7004，曲线数 168/180，全面超 512c 基线（0.6741/0.6512）
- 管线：extractor 单一入口，segmenter: cv|unet|multi_unet；structure_backend: cv|yolo；ocr: stub|paddle

## 〇、先读这些（每次会话开始必读）

1. **`F:\CODE\New\Prompt.md`**（只读，禁止修改）— 项目唯一权威指导文件（竞赛一等奖目标、六大要求、验收标准）。
2. **`F:\CODE\New\baseline\README.md`** — baseline 完整架构/用法/评估/限制。
3. **`F:\CODE\New\baseline\C_DESIGN.md`** — Phase C 设计+四轮训练记录+失败分析（已补充根因与修复记录）。
4. **`F:\CODE\New\baseline\C_RESEARCH.md`** — **2026-08-19 三路并行调研汇总**（25+ 来源：LineFormer/ChartZero/GOI/clDice/EpiCurveBench/Graph-FINDER/STU-Net/Focal Tversky 等，含证据×成本方案矩阵）。
5. **`F:\CODE\New\baseline\C_FIX_DESIGN.md`** — 根因修复设计+重训方案+后续候选。
6. **`F:\CODE\New\baseline\PHASE_D_ACCEPTANCE.md`** — Phase D 验收材料清单（验收集/评估命令/ablation 矩阵）。
7. **`F:\CODE\New\baseline\AXIS_TEXT_RESEARCH.md` / `B5A_DESIGN.md` / `TESTING_GUIDE.md` / `web\README.md`** — 历史设计/测试/Web 文档。
8. 开始工作前：`cd F:\CODE\New\baseline && git log --oneline -15 && git status`。

## 一、项目背景与验收标准（摘要）

**主题：** 材料科学图像曲线智能识别与解析（自动从曲线图提取数据 → 结构化输出）。
**验收：** 曲线数据点 RMSE ≤ 坐标轴满量程 1%；坐标类型判断准确率 ≥98%；单张 <5s（GPU）；多曲线召回 ≥95%（Phase C）。
**验收方式：** ① dataset-platform 独立 seed 500 张测试图（已生成 500 多曲线 + 500 单曲线）；② ≥50 张真实论文蠕变图人工标注对比；③ 每个模块 ablation。
**首批领域：** 蠕变曲线（做透后再泛化）。

## 二、环境与仓库状态

**环境：** Anaconda 虚拟环境 **`mci`**（Python 3.11，RTX 4060 Laptop 8GB / CUDA 12.6）
- torch 2.13.0+cu126（GPU ✓）、ultralytics 8.4.115（YOLOv8n）、paddlepaddle 3.3.1（CPU）+ paddleocr 3.7.0
- numpy **必须固定 1.26.4**；fastapi 0.141.1 / uvicorn 0.52.3
- **租 GPU 选项（用户已暂缓）**：AutoDL 3090/4090 24GB（¥1.2-2.5/h）可显著加速大模型/高分辨率实验；代码 0.46MB + 数据 923MB 迁移成本低；云端 Linux 无 torch/paddle DLL 冲突

**仓库：** `F:\CODE\New\baseline`（git，master；2026-08-19 会话 8 个新 commit）
**数据工厂：** `F:\CLAUDE\NewProject1\materials-curve-dataset-platform`（V0fix-final-2，平台仓库零改动）

**数据目录（均不入库，gitignore）：**
| 目录 | 内容 |
|------|------|
| data/train_platform(+_4c/_5c/_single) | 2000+600+300+600 张训练集（2-5 曲线） |
| data/val_multi / val_single | 120 张 4 曲线 + 60 张单曲线验证集（Phase C 评估基准） |
| **data/eval_phased_500 / _single** | **Phase D 验收集（各 500 张，独立 seed 20260819，已生成）** |
| data/eval_multi_diag_512c | 逐曲线失败诊断（diag.json + error_pattern.json + vis/） |
| data/real_papers/ | 真实论文图收集区（**待用户**） |
| data/failures/ | 真实失败样本库 |

**模型检查点（models/checkpoints/）：**
- `unet_curve.pt` — 单曲线 U-Net 512（val_iou 0.8166，默认）
- **`unet_multi_curve_evalfix_r2.pt` — 多曲线 K=6，重训完成（4+21 epoch），val_iou 0.7636（当前最佳）**
- `unet_multi_curve_evalfix.pt` — epoch 4 暂停点（0.7345，保留为安全基线）
- `unet_multi_curve_512c.pt`（旧最佳 0.4993）/ `_512d.pt` / `v3.pt` 等历史版本
- `models/detection/yolo_struct.pt` — YOLOv8n 结构检测（mAP50 0.942）

## 三、当前进展（2026-08-19 会话新增）

### 3.1 根因：训练目标与评估基准不一致（本会话最重要发现）
- 训练侧 `_fit_axis_from_labels` 的 y 标签过滤 `c[1] < img_h-85` 排除最底部 y 刻度（如 0.001@527px，600px 图），剩余序列 0.1/1/10 比值=100 不触发 log 判定（阈值 >100）→ 判 linear；评估侧 `_classify_labels` 保留底部标签 → 判 log。
- **全数据核查：y 轴不一致 904/3548（25.5%）、x 轴 65/3680（1.8%），全部为 fit=linear vs GT=log**。
- log 轴上 linear 拟合误差可达 100+px → 25% 训练图掩码系统性错误 → 训练平台化、systematic_bias（同图 4 曲线同向偏 -1.4%）、log 轴图全失败（20/20 掩码差异 >8000px）、单曲线 4/5 失败图同根因（img_0036 训练掩码全零）。

### 3.2 修复（已实施并验证，git e3a2d79）
- ChartDataset 改走**与评估完全相同的代码路径**：detect_structure → read_ticks → build_axes → value_to_pixel 重建实例掩码（12ms/图，per-image 缓存；旧 polyfit 保留为 fallback）。
- 验证：240/240 val_multi 掩码通道与 GT 曲线像素重合 ≥95%（≤2px）；pytest 106/106（后 110/110）。
- 训练脚本增强：EMA（--ema-decay 0.999）+ 骨架召回损失（GT 骨架预计算缓存，随增强同步）；--init 现在也转移 out 头（K=6 续训不再随机初始化输出层）。

### 3.3 重训（暂停在 epoch 4）
- 命令：train/train_segmentation_multi.py --data-dir 4 个训练目录 --val-dir val_multi,val_single --epochs 25 --batch 8 --size 512 --per-dir-limit 600 --init ...512c.pt --ema-decay 0.999
- 进度：epoch 1 val_iou 0.6763 → epoch 4 **0.7345**（旧 512c 最终 0.4993，+47%）；约 8-10min/epoch；25 epoch 全程 ~3-4h。**恢复命令见第八节**。
- 注：曾有一次因 init 未转移 out 头而重启（epoch 1 仅 0.056 被识别并修正）。

### 3.4 评估/诊断工具（新增，已入库）
- `scripts/eval_multi_6ab.py` — LineFormer 式 6a/6b 双口径（6a 纯召回、6b 罚虚检）；基线 6a=0.6767/6b=0.6534
- `scripts/eval_multi_diag.py` — 逐图/逐曲线 rel_rmse + 失败桶分类 + 模板/形态归因
- `scripts/diag_error_pattern.py` — 错误模式分类（local_spike 91 / systematic_bias 34 / mixed 39 / partial_trace 3）
- `scripts/diag_pixel_bias.py` / `diag_prob_peak.py` — 像素级偏差/模型概率峰值偏移（+1~2px）
- `scripts/diag_mapping_consistency.py` — 训练/评估映射一致性（复现根因核查）
- `scripts/diag_ghost_*.py` / `exp_argmax_vs_indep.py` — 虚检通道特征与 argmax 互斥实验

### 3.5 其他完成项
- **legend_matcher 增强**：颜色距离关联曲线标签（annotation-only 不丢曲线）+ 4 测试；extractor 传 image_bgr
- **Web 多曲线**：segmenter 增加 multi_unet 选项（前后端+配置）；configs/baseline.yaml 增加 multi_* 参数
- **min_area 虚检抑制**：curve_extractor 增加 multi_min_area 配置（曲线数 47/60→57/60，6b +1.5pp）
- **Phase D 验收集**：500 多曲线（曲线数 1-5，轴型 4 组合）+ 500 单曲线，独立 seed
- **argmax 实验结论**：曲线交叉重叠区占图内墨迹 10-32%（img_0007 31.6%），argmax 互斥在交叉处丢弃一个通道的像素 → local_spike 结构性来源；独立阈值多标签平均多保留 2497px/图（LineFormer 方向）

### 3.7 2026-08-20 云端实验轮（AutoDL 4090，已关机）

1. **增强 flip bug 修复**（commit 9f7c66c，重大）：cv2.flip 对 (K,H,W) 掩码——512 下 OpenCV 绑定按通道启发式只翻高度轴（垂直镜像），与图像水平镜像错位（历史全部训练 50% 样本目标污染）；768 下直接崩溃。修复 np.flip(inst,2)[::-1] + 2 回归测试。
2. **768 分辨率实验**（unet_multi_768.pt）：val_iou 0.7531；评估 6a=0.6963/6b=0.6861/**曲线数 172/180（最高）**——无端到端收益（后处理像素阈值尺度敏感）
3. **512-fixed 对照**（unet_multi_512fix.pt）：val_iou **0.7714（历史最高）**、loss 0.145（buggy 训练 0.573 的 1/4）；评估 6a=0.7093/6b=0.6964/170/180——增强修复改善 val_iou 但 6a/6b 中性
4. **结论**：①分辨率 768 不投；②像素指标与端到端召回再次脱节；③**最佳组合仍为 evalfix_r2 + ind + ma450（6a 0.7185/6b 0.7004）**；④云端流程全自主化已验证（SSH/装环境/传数据 1.2G/训练/评估/下载，见第十节）

### 3.6 2026-08-20 会话：重训完成 + 评估 + 方案 C Phase 1（commit 73fe133 / 51dc66e）

1. **重训完成**：evalfix.pt（epoch 4）续跑 21 epoch（共 25）→ **evalfix_r2.pt，val_iou 0.7636**
   （+53% vs 512c 0.4993；+4% vs 暂停点 0.7345）。注意：LR 余弦重启造成 epoch 2/7/15
   val_iou 回落（最低 0.4107，瞬态 EMA 伪影），best-iou 保存策略保证 checkpoint 只升不降；
   epoch 17-21 连续新高收敛（lr→0）。教训：**从暂停 checkpoint 续训会经历 LR 重启回退，
   不必干预，EMA + best-iou 兜底**。
2. **评估发现回归并修复**：evalfix_r2 + argmax 6a/6b = 0.6074/0.5889 < 512c 基线
   （当前数据重跑 0.6741/0.6512，注意历史基线 n_gt=532 与现在 540 不同口径，须重跑对比）。
   逐图定位：回归集中在交叉密集区（marker_rich/multi_curve_comparison/three_stage 模板），
   新模型交叉区双通道高概率 → argmax 互斥更伤（img_0069 列级验证）；29 条转好（log 轴/
   systematic 修复生效）、61 条转差。
3. **顺带修复 _truncate_jumps 越界 bug**（陡尾链 IndexError，短路掩盖；新模型 7 图触发）+
   3 回归测试，pytest 113/113。
4. **方案 C Phase 1 实施（multi_independent_mask）**：交叉处去 argmax 互斥（LineFormer
   多标签方向）。val 集：argmax 0.6074/0.5889 → +independent 0.7222/0.6667 →
   **+min_area450 = 0.7185/0.7004，曲线数 168/180——三项全面超基线**。450-550 min_area
   结果相同（虚检面积分布有间隙）。
5. **Phase D 验收**（独立 seed 20260819，500+500）：多曲线 6a=0.7869/6b=0.7133/346/500；
   单曲线 500/500 全过，rel_rmse 中位 0.18%、p90 0.40%、max 1.77%、≤1% 达标率 99.4%。
6. **配置更新**：configs/baseline.yaml 默认 multi_unet_checkpoint=evalfix_r2.pt、
   multi_independent_mask=true、multi_min_area=450（Web 演示自动生效）。
7. 剩余差距：6a 距 95% 目标仍远（1-5 曲线独立集）；虚检（6b 0.7004）与曲线数准确率
   （346/500）是主要扣分项；候选：Phase 2 分支配对（浅角交叉）、GOI 损失（E）、
   min_area 自适应、真实图验证（待用户图）。


## 四、测试与质量
- pytest **115/115**（+4 legend_matcher、+3 _truncate_jumps、+2 增强 flip 回归）
- 单曲线回归基准（不得回退）：合成 med ≤0.40%/87.5%、平台 100% ≤1%、paddle 99%
- 多曲线基线（当前数据重跑口径 540 GT）：512c 6a=0.6741/6b=0.6512/161/180；**evalfix_r2+independent+min_area450：6a=0.7185/6b=0.7004/168/180（2026-08-20 最佳）**；Phase D 500 图：6a=0.7869/6b=0.7133/346/500

## 五、待用户任务
- **收集 ≥50 张真实论文蠕变图**（先 10-15 张）→ data/real_papers/raw/ + gold 标注；失败图放 data/failures/ 或发路径
- ~~决定重训策略~~（2026-08-20 已执行：evalfix epoch 4 续跑 21 epoch 完成，val_iou 0.7636）

## 六、已知的技术坑（务必先读，含本会话新增）

1. **Windows DLL 冲突**：torch 与 paddle 同进程互斥（WinError 127）。OCR 用 CPU paddle（enable_mkldnn=False）；PaddleOCRBackend._ensure 先 import torch 再 import paddle。
2. **numpy 固定 1.26.4**（mci 环境）。
3. **matplotlib 渲染差异**：GT 像素用 `fig.canvas.buffer_rgba()`，勿用 savefig 反推。
4. **conda run 偶发插件问题**——用 `F:\anaconda3\envs\mci\python.exe` 直接调用。
5. **paddle 评估慢**：~30s/图 → 评估期间勿并行 YOLO/U-Net 训练。
6. **'10N' 上标粘连**：resolve_values 序列一致性重解析（勿回退 B-5a 候选消歧）。
7. **训练目标必须与评估基准一致（本会话最大教训）**：不仅 CSV 重建掩码，**轴映射代码路径也必须一致**——训练侧任何标签过滤/拟合差异都会造成 log 轴整体错位（25.5% 训练图曾受害）。修改训练数据管线后必跑 scripts/diag_mapping_consistency.py 核查。
8. **U-Net 推理分辨率必须等于训练分辨率**（512；--unet-size 切换）。
9. **log 轴 1px≈2% rel 误差**：log 轴图上像素级偏移（模型概率峰值 +1~2px）会被放大为 1-2% 边缘失败。
10. **曲线交叉/靠近**：argmax 互斥丢弃交叉处一个通道像素（重叠区 10-32% 图内墨迹）→ local_spike；候选修复：独立阈值多标签 + 方向连续性追踪（LineFormer 式）。
11. **虚检通道**：pred=5 时第 5 通道面积 51-625px（小碎片）或与真曲线同量级（过分割）——min_area 过滤对小碎片有效（6b +1.5pp），过分割需 NMS/IoU 合并。
12. **训练脚本 init 需转移 out 头**：从 K=6 checkpoint 续训时若丢弃 out 层，输出头随机初始化（epoch 1 val_iou 仅 0.056）；已修复（--init 保留 out.* 权重）。
13. **PowerShell 重定向缓冲**：`*> log 2>&1` 时 Python print 全缓冲，日志可能 0 字节——监控训练用 checkpoint 时间戳/val_iou 字段，或 python -u。
14. **gitignore 陷阱**：runs/ 与 *.pt 已 ignore；data/ 全量不入库。
15. **glob 过滤**：evaluate/run_baseline 必须过滤 `*_mask.png`。
16. **PaddleOCR 长条带**：超长条带被 resize 到 max_side_limit 4000 内变形——裁剪要合理。
17. **cv2.flip 三维数组陷阱（2026-08-20 新发现，重大）**：cv2.flip(inst, 2) 对 (K,H,W) 掩码——Python 绑定按「末维 ≤512 视为多通道图」启发式处理：512 分辨率时只翻转高度轴（垂直镜像，通道保留），与图像的水平镜像错位（历史全部训练 50% 样本目标被污染，疑似训练平台化/0.76 天花板元凶之一）；768 分辨率时（末维 >512）直接断言崩溃。已修复：numpy np.flip(inst,2)[::-1]（水平镜像+通道顺序反转，与图像一致），+2 回归测试（commit 9f7c66c）。注意：修复后模型与全部历史 checkpoint 的训练分布不同，对比须注明。

## 七、工作约定（继承 Prompt.md 六大要求）

- 每次会话开始重读 `Prompt.md` + 本文件 + 相关设计文档 + git log/status；
- **（用户强调）无论何时先做深入研究再动手**：每项工作开工前 ≥1 轮 web_search/论文/社区/官方文档调研，形成设计依据（可参考 C_RESEARCH.md 的证据×成本矩阵），先给用户确认设计再实现；
- 每步改动跑 `python -m pytest tests -q` 与 `scripts/evaluate.py --ocr stub` 回归，**指标不得回退**（stub 基准：合成 med ≤0.40%/87.5%、平台 ≤1% 达标率 100%）；
- 保持「模块接口稳定、逐步替换实现」；`data/` 不入库（代码例外），模型与训练数据本地；
- 定期归档：重大里程碑后复制 `baseline` 为 `baseline_backup_<日期>`（含 .git）。

## 八、常用命令速查（工作目录 F:\CODE\New\baseline）

```bash
# 注意：优先用 F:\anaconda3\envs\mci\python.exe（conda run 偶发插件问题）
PY=F:\anaconda3\envs\mci\python.exe

# 【首选】恢复 C-2 重训（续跑 25 epoch 或更多）
$PY train/train_segmentation_multi.py --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single --val-dir data/val_multi,data/val_single --epochs 25 --batch 8 --size 512 --per-dir-limit 600 --init models/checkpoints/unet_multi_curve_512c.pt --out models/checkpoints/unet_multi_curve_evalfix.pt --ema-decay 0.999

# 多曲线评估（6a/6b 双口径，推荐）
$PY scripts/eval_multi_6ab.py --data-dir data/val_multi,data/val_single --model models/checkpoints/unet_multi_curve_evalfix.pt --out-dir data/eval_multi_6ab_evalfix --size 512

# 失败模式诊断（逐曲线桶分类 / 错误模式 / 像素偏差）
$PY scripts/eval_multi_diag.py --model models/checkpoints/unet_multi_curve_evalfix.pt --out-dir data/eval_multi_diag_evalfix --size 512
$PY scripts/diag_error_pattern.py --model models/checkpoints/unet_multi_curve_evalfix.pt

# 单曲线评估（stub 快 / paddle 慢 ~50min/100 张）
$PY scripts/evaluate.py --data-dir data/eval_phased_500_single --out-dir data/eval_phased_single --ocr stub --segmenter unet

# Phase D 多曲线验收（500 张独立验收集）
$PY scripts/eval_multi_6ab.py --data-dir data/eval_phased_500 --model models/checkpoints/unet_multi_curve_evalfix.pt --out-dir data/eval_phased_multi --size 512

# Web 演示
$PY web/app.py                      # http://127.0.0.1:8000

# 数据生成
$PY data/dataset_builder.py --out-dir data/eval_xxx --count 100 --num-curves 1 --seed <新seed>

# 测试
$PY -m pytest tests -q              # 当前 110/110
```

## 九、待用户确认/执行的事项

1. **真实论文图收集**（Phase A.3）：≥50 张蠕变图，先 10-15 张 → data/real_papers/raw/；
2. **重训策略**：恢复训练至 25 epoch / 用 epoch 4 checkpoint 先评估 / 调整超参后再训；
3. **验收口径**：95% 多曲线召回在什么数据集上考核（合成独立 seed？真实图？含曲线数错误惩罚否——6a/6b 双口径已备）；
4. **GPU 租用**（已暂缓）：后续大模型/高分辨率实验可租 AutoDL 4090（¥2-2.5/h，本地 8GB 受限）；
5. 收集失败图（Web 上传测试失败）是回归样本的重要来源。

> 交接 v5 完成于 2026-08-19。上一版 v4（2026-08-18）见 git 历史。
> 详细研究依据见 C_RESEARCH.md；根因修复设计见 C_FIX_DESIGN.md；验收材料见 PHASE_D_ACCEPTANCE.md。

## 十、AutoDL 云端自主工作流（2026-08-20 起，AI 自主执行）

### 10.1 实例与环境（已就绪）
- **027 机 RTX 4090 24GB**（¥1.98/h 按量；040 机同配置 ¥2.18/h），Ubuntu 22.04
- SSH：见 `F:\CODE\New\autodl_secret.json`（**不入库**，密码变更只改此文件）；助手脚本 `F:\CODE\New\autodl.py`（run / upload / upload_dir）
- 环境：conda base Python 3.12.3；torch 2.8.0+cu128（先验证兼容，不兼容再重装 2.13.0+cu126）；**numpy 1.26.4 已固定**；paddlepaddle 3.3.1 + paddleocr 3.7.0（CPU 版即可）；ultralytics 8.4.115；fastapi/uvicorn 已装
- 代码：`/root/baseline`（已解压，含全部配置/脚本/文档）；数据：`/root/baseline/data`（1.1GB，已含训练/验证/Phase D 全套）；模型：`/root/baseline/models/checkpoints`（evalfix_r2 / evalfix / 512c / unet_curve / yolo_struct）

### 10.2 AI 可自主完成的环节（已验证）
1. SSH 连接（paramiko 密码登录）
2. 环境检查 + 依赖安装（版本与本地 mci 完全一致，numpy 1.26.4 为硬约束）
3. 代码/数据/模型上传（SFTP；1.1GB 数据 + 125MB checkpoint 一次性传输）
4. import 冒烟测试（torch/numpy/paddle/ultralytics/mci 全通过）
5. 训练执行与监控（有卡开机后：`--batch 16-32`、25 epoch 预计 <1h、best-iou checkpoint 自动保存）
6. 评估执行（eval_multi_6ab / evaluate.py / diag 全套）
7. 结果下载回本地（SFTP）

### 10.3 必须用户操作的环节（唯一硬依赖）
1. **有卡开机/关机**：AutoDL 控制台点"开机"（无卡→有卡，可能排队；4090 空闲 1/8）；或提供 AutoDL API token 后可由 AI 调 API 开机（未实现，待用户决定）
2. **账户余额**：按量计费从账户扣款
3. 关机提醒：训练/评估完成后 AI 会提醒用户关机（或配置 API 自动关机）

### 10.3.5 云端 768 分辨率实验（2026-08-20 进行中）
- 动机：1-2% 边缘失败桶最大（87 条）；C_RESEARCH「分辨率是细线任务首要杠杆」
- 命令：--size 768 --batch 8 --init evalfix_r2.pt --out unet_multi_768.pt --epochs 25（显存 17.7GB/24GB）；已修复增强 flip bug 后启动（commit 9f7c66c）
- 对照计划：768 完成后跑 512-fixed（同修复、同 init）分离分辨率 vs 修复增益
- **768 结果（完成）**：val_iou 0.7531（loss 0.153 vs 512 的 0.573，修复后目标一致）；评估（--size 768，ind+ma450 配置）：6a=0.6963/6b=0.6861/**曲线数 172/180（历史最高）**
- **512-fixed 对照（完成）**：val_iou **0.7714**（历史最高，loss 0.145）；6a=0.7093/6b=0.6964/曲线数 170/180
- **三模型结论（2026-08-20）**：① 分辨率 768 无端到端收益（6a 反而 -1.3pp，后处理像素阈值尺度敏感）；② 增强修复大幅改善 val_iou/loss 但 6a/6b 中性（再次印证像素指标与端到端脱节）；③ **当前 6a/6b 最佳组合 = evalfix_r2 + independent + min_area450（0.7185/0.7004）**，曲线数最高 = 768-fixed（172/180）

### 10.4 云端训练命令（开机后直接用）
```bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/baseline
# 方案 C 验证重训（batch 16-32，25 epoch，预计 <1h；可加 --init evalfix_r2.pt 续训）
python train/train_segmentation_multi.py --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single --val-dir data/val_multi,data/val_single --epochs 25 --batch 16 --size 512 --per-dir-limit 600 --init models/checkpoints/unet_multi_curve_512c.pt --out models/checkpoints/unet_multi_curve_cloud.pt --ema-decay 0.999
# 评估（6a/6b）
python scripts/eval_multi_6ab.py --data-dir data/val_multi,data/val_single --model models/checkpoints/unet_multi_curve_cloud.pt --out-dir data/eval_multi_6ab_cloud --size 512
# Phase D 验收
python scripts/eval_multi_6ab.py --data-dir data/eval_phased_500 --model models/checkpoints/unet_multi_curve_cloud.pt --out-dir data/eval_phased_multi_cloud --size 512
```

### 10.5 云端 vs 本地分工
- **云端**：重训（E/H 方案：base 96/128、768-1024 分辨率）、全量验收评估——显存/速度优势
- **本地**：日常开发、小实验、pytest、真实图验证、文档
- 注意：云端训练脚本 base=64 硬编码（方案 H 需参数化，改 train_segmentation_multi.py 的 UNet(base=64) 为 --base 参数，改动前先本地验证）

