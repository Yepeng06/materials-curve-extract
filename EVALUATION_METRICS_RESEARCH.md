# 曲线提取效果评价指标：文献调研与权威指标确定（2026-08-26）

> 任务：确定一个/几个**权威的评价指标**（评价曲线提取的效果），优先采用科研论文主流指标。
> 调研方式：4 路并行研究子代理（web_search 深度检索）+ 本人对关键论文**一手来源**的全文/官方代码抓取核实：
> LineFormer 论文全文与官方评估代码、CHART-Info 官方指标定义（adobe-research/CHART-Synthetic 仓库 metric6b.tex + metric6a.py）、
> EpiCurveBench（arXiv 2605.27195 全文）、ChartZero（arXiv 2605.05820 全文）、ChartRecover（Nature Comm. Eng. 2026 全文）、
> Graph-FINDER（npj Comput. Mater. 2026）、ChartQA 官方评估代码。
> 证据等级：A=官方代码/竞赛规则/论文全文；B=预印本/官方仓库；C=社区实践。

---

## 〇、结论先行（只读这一节）

**推荐指标集 = 3 个主指标 + 4 个辅助/诊断指标，全部有主流论文背书，且与 Prompt.md 验收标准逐条对齐：**

| 层次 | 指标 | 文献出处（权威性） | 对应验收标准 |
|---|---|---|---|
| **主 1** | **6a 召回率**（曲线级 rel-RMSE ≤1% 判成功，Hungarian 匹配，分母 N_gt） | CHART-Info 竞赛（ICDAR'19/ICPR'20）→ LineFormer（ICDAR'23）→ ChartZero（2026）沿用 | 多曲线召回率 ≥95% |
| **主 2** | **6b 召回率**（同上但分母 max(N_gt,N_pred)，虚检记 0 分） | 同上（LineFormer 论文公式 K=max(Ng,Np)） | 鲁棒性展示（虚检惩罚） |
| **主 3** | **nRMSE 分布**（每曲线 RMSE ÷ 轴满量程，报中位/p90/最大/≤1% 达标率） | ChartZero（RMSE/Range(Y)）、Graph-FINDER（nRMSE=0.018）、本项目验收口径 | 曲线点 RMSE ≤ 满量程 1% |
| 辅助 1 | 曲线数准确率（N_pred=N_gt 的图占比） | 本项目现有；PseCo（CVPR'24）计数一致性思路 | — |
| 辅助 2 | 官方 CHART-Info **连续 6a/6b**（逐点相对误差 F1 版） | CHART-Info 官方 metric6a.py（LineFormer 即用此代码报数） | 与文献数值可比 |
| 辅助 3 | 错误桶分布（≤1%/1-2%/2-5%/>5%）+ 模板/轴型分层 | 本项目现有；EpiCurveBench 误差分解思路 | 失败模式诊断 |
| 辅助 4 | IoU / clDice（像素级，训练/分割诊断） | ChartZero（IoU 0.82）、clDice（arXiv:2003.07311） | 分割质量诊断 |

**可选项（真实图阶段/论文加分）**：ECS（EpiCurveBench 2025，时序对齐质量）、ChartRM（ChartZero 2026，端到端门控）、Spearman ρ（Graph-FINDER，形状一致性）。

**三个必须修的口径问题（详见 §三）**：
1. **匹配算法**：现状为贪心匹配 → 应改 **Hungarian/线性分配**（官方实现 scipy.optimize.linear_sum_assignment，接近区多曲线时贪心会错配）；
2. **归一化分母**：现状用"GT 曲线 y 跨度" → 验收口径是"**轴满量程**"（合成数据 meta.json 的 y_range 可直接取）。⚠️ **实证结论与直觉相反**：满量程归一更宽松（分母=整轴范围，通常大于曲线跨度），跨度归一更严格——val 540 条曲线实测：span 版召回 85.0% vs 满量程版 **88.5%**，19 条曲线因换分母从失败翻转为通过（反向 0 条），中位 rel 缩小 40%；
3. **x 覆盖**：现状只统计重叠区（截断预测可蒙混过关）→ 应加"未覆盖 GT 段按满误差计"（官方 CHART-Info 连续指标天然惩罚截断）。

---

## 一、文献指标地图（按指标族，全部注明一手出处）

### A. 端到端数据提取指标（曲线级 · 数据坐标域）—— 本项目的"主战场"

#### A1. CHART-Info 6a/6b —— 折线图数据提取的事实标准 ⭐

**出处**：ICDAR 2019"Harvesting Raw Tables from Infographics (CHART-Info)"竞赛（Adobe Research + UB）提出，ICPR 2020 CHART-UB 沿用；
**LineFormer**（ICDAR'23, arXiv:2305.01837）以实例分割范式重做 line chart 提取并沿用 6a/6b 报数；ChartZero（2026）继续引用该口径。

**一手来源**（本报告已抓取核实）：
- 官方定义：`adobe-research/CHART-Synthetic` 仓库 `metric6b.tex`（连续型数据序列指标公式）
- 官方实现：LineFormer 仓库 `metric6a.py`（`metric_6a_indv` / `metric_6b_indv`，即 CHART-Info 官方代码）
- LineFormer 论文 §7 公式：score = 1/K · max_X Σᵢⱼ SᵢⱼXᵢⱼ，K=N_g（6a）/ max(N_g,N_p)（6b）

**精确定义（连续曲线）**：
- 逐点误差：`Error(vᵢ,uᵢ,P) = min(1, |vᵢ − I(P,uᵢ)| / (|vᵢ| + ε))`，其中 I(P,uᵢ) 为预测曲线在 GT 点 x=uᵢ 处的插值，**ε = GT 曲线 y 范围/100**（1% 量程兜底）；
- 召回：`Recall(P,G) = (1/Σinterval) · Σᵢ (1−Error(vᵢ,uᵢ,P))·Interval(i)`，Interval 为 GT 点的 x 区间权重（首尾半区间、中间全区间），即**按 x 弧长加权的逐点相似度均值**；
- 精度：`Precision(P,G) = Recall(G,P)`（角色对调）；配对得分 = **F1 = 2PR/(P+R)**；
- 多曲线：pairwise 相似度矩阵 + **Hungarian 最优分配**；6a 分母 = N_g（纯召回，未匹配 GT 记 0）；6b 把矩阵补方到 K=max(N_g,N_p)，未匹配预测记 0 分并放大分母（罚虚检）；
- 聚合：**逐图打分后对图取平均**（每图等权）；
- 6a 与 6b 的关系：官方竞赛中 6a 跑**像素域**（'visual elements'，视觉元素检测/分类），6b 跑**数据域**（'data series'，语义值提取，另含图例名匹配 β=2）。⚠️ **口径澄清（子代理 A 代码级核实）**：LineFormer 仓库 eval.py 对 6a、6b **都在像素空间**计算（读 'visual elements'，无图例名匹配）——其论文报告的 6b（UB-PMC 88.25）与竞赛官方 6b 的数据值口径**不完全等价**；本项目评估应直接用官方 metric6b.py 的**数据值坐标口径**（跨数据集可比、log 轴语义正确）。

**报告数值**（LineFormer 论文 Table 1，一手核实）：
| 方法 | AdobeSynth19 6a/6b | UB-PMC22 6a/6b | LineEX 6a/6b |
|---|---|---|---|
| ChartOCR | 84.67 / 55 | 83.89 / 72.9 | 86.47 / 78.25 |
| LineEX | 82.52 / 81.97 | 50.23 / 47.03 | 71.13 / 71.08 |
| **LineFormer** | **97.51 / 97.02** | **93.1 / 88.25** | **99.20 / 97.57** |

（注意各方法 6b 相对 6a 的掉幅：UB-PMC 上 ChartOCR 掉 11pp、LineEX 掉 3.2pp、LineFormer 掉 4.85pp——6b 对虚检的敏感性一目了然，这正是双口径的价值。）

**点评**：
- ✅ 行业标杆（ICDAR'19 至今 7 年），与 LineFormer/ChartZero 直接可比；连续得分（0-1）比二值化召回更精细；6b 天然惩罚虚检。
- ⚠️ 逐点误差是**值相对**（除以 |vᵢ|）——对跨数量级的数据（如 log 轴小值端）误差会爆炸（有 ε 兜底但仍敏感）；**与"RMSE ≤ 1% 满量程"不是同一个量纲**。
- ⚠️ np.interp 在预测范围外**钳位到端点值**（不平滑外推），所以截断预测会被大误差惩罚——这是优点（惩罚截断）。

#### A2. 阈值化召回率家族 recall@τ —— 可解释、对接验收 ⭐

**出处**：FigureSeer（ECCV'16）与 Linear Programming（WACV'22，Kato et al.）**逐点误差 < 2% 判为正确**（LineFormer 论文 §7 原文确认）；ChartRecover（Nature Communications Engineering 2026, s44172-026-00691-8）推广为多阈值。

**ChartRecover 的精确定义（一手全文核实）**：
- 相对误差：`ε_v = |v_pred − v_gt| / (|v_gt| + ε)`，ε=1e-8，v∈(x,y)；
- **Accuracy(τ) = (1/N_gt) Σᵢ [ε_vi < τ] × 100%，τ ∈ {2%, 5%, 10%}**；
- 匹配：先按 Metric6a 距离矩阵跑 **Hungarian**，再统计阈值内配对占比。

**同族最贴近的先例——Liu & Klabjan（arXiv:1906.11906，"Data Extraction from Charts via Single Deep Neural Network"）**：
- `Error = |Value_pred − Value_GT| / |Value_GT|`（式 8），按 **<1%/5%/10%/25% 阈值**统计 Accuracy；另有 **"ALL" 硬判据**：标题、轴标签全对且所有元素值误差 <1% 才计正例；
- 报数（条形/饼图集）：Simulated ALL 79.4%、Tuple 1% 误差 81.0%；FigureQA ALL 69.2%——**"1% 相对误差阈值 + 全图达标才计正例"正是本项目 RMSE≤1% 验收思路的最直接文献先例**（虽非折线，公式可直接沿用）。

**本项目现状（同族变体）**：曲线级 rel-RMSE = RMSE ÷ GT 曲线 y 跨度，≤1% 判"召回"，6a/6b 双口径 + 贪心匹配（`scripts/eval_multi_6ab.py`）。

**点评**：
- ✅ 与"召回率 ≥95%、RMSE ≤1%"的验收语言天然一致，答辩/竞赛可解释性强；ChartRecover（Nature 系期刊）用同族指标，权威性有背书。
- ⚠️ 二值化丢失"差多少"信息（1.1% 与 50% 同为失败）→ 必须配 nRMSE 分布；⚠️ 阈值口径（1% vs 2%）不统一时数值不可比 → 报告必须写死定义。

#### A3. NRMSE / rel-RMSE（量程归一）—— 数值精度主指标 ⭐

**出处与定义**：
- **ChartZero**（arXiv:2605.05820, 2026）：`NRMSE = RMSE(y, ŷ) / Range(Y)`（Y 为轴 y 范围，从 VLM 提取的轴元数据；注意正文未给出 NRMSE 分母公式原文，Range(Y) 出自同文 ChartRM 的 Γ_data = exp(−λ·RMSE/Range(Y))），消融报 0.071→**0.028**。
- **Graph-FINDER**（npj Computational Materials 2026, s41524-026-02247-y）：**nRMSE = RMSE ÷ 参考曲线的 y 轴动态范围**（图注原文 "normalized pointwise-error metric relative to the reference y-axis dynamic range"），mean nRMSE = **0.018 ≈ 1.8% 满量程**；配 **Spearman ρ > 0.92**（形状一致性）双指标，明确"形状/数值解耦"。
- **本项目**：`src/mci/eval/metrics.py` 的 rel_rmse（当前除以曲线 y 跨度；验收标准"RMSE ≤ 坐标轴满量程 1%"要求除以轴满量程）。
- ECS（EpiCurveBench）的替换罚也按**曲线自身 y 范围**归一（θ=0.01）——同一族。

**点评**：
- ✅ 量程归一与验收标准同构，log 轴友好（1px≈2% 的物理极限直接可读）；ChartZero/Graph-FINDER 双双采用，材料/图表领域权威性足。
- ⚠️ 归一化分母必须写死：**轴满量程**（验收）vs **曲线跨度**（现状，ECS/Graph-FINDER 同族；实证：平缓曲线用跨度归一更严格，见 §三.2）结果差异大——建议双报。
- ⚠️ RMSE 是 L2，单点大误差会拖均值 → 报分布（中位/p90/最大）而非只报均值。

#### A4. ECS / ERP（EpiCurveBench 2025）—— 时序对齐质量（可选项）⭐新

**出处**：EpiCurveBench（medRxiv 2025 → arXiv:2605.27195 v2），面向"真实曲线图数字化质量"提出，**2025-2026 最新权威指标**。

**精确定义（一手全文核实）**：
1. **系列匹配**：Normalized Levenshtein Similarity（NLS = 1 − d_L/max(|s₁|,|s₂|)），阈值 0.5，图例名/列名配对；
2. **对齐**：ERP（Edit Distance with Real Penalty，Chen & Ng 2004）动态规划递推
   `C(i,j) = min{ C(i−1,j−1)+D_θ(p_i,t_j);  C(i−1,j)+λ;  C(i,j−1)+λ }`，
   边界 C(0,0)=0、C(i,0)=i·λ、C(0,j)=j·λ，**gap 罚 λ=1**（显式惩罚截断/缺失段——DTW 没有 gap 罚，截断可蒙混）；
3. **替换罚**：`δ_ij = |p_i − t_j| / (y_max − y_min)`（按该序列**自身 y 范围**归一），
   `D_θ(p_i,t_j) = min(1, δ_ij/θ)`，**容差 θ = 0.01**（默认；差异 <1% 量程按比例罚，≥1% 记全错——与本项目"1% 桶"同构）；
4. `ECS(p,t) = 1 − C(M,N)/(#matches + #gaps)`；
5. 逐图对匹配序列取均值，**未匹配 GT 序列记 0**；再对全数据集平均。

**报告数值**：最强 VLM（Gemini 2.5 Pro）仅 **52.3% ECS**；标注者间 ECS 90.9%（可用性上界）；ECS 与下游四项统计误差的 Spearman 相关是 DTW 的 **1.5-3.6×**（gap 罚的价值实证）。

**点评**：
- ✅ 专为"曲线数字化质量"设计：容忍小幅时序偏移、比例化惩罚、显式 gap 罚；对"提取结果是否可用"（下游统计）有实证验证。
- ⚠️ **虚检/漏检行为（精确版）**：整条虚检系列（NLS 匹配不上任何 GT）被忽略不计；**整条漏检系列记 0 分**；但**单点虚检走"插入"分支（每点罚 λ=1 计入分子分母）、单点漏检走"删除"分支**——"虚检不罚"仅对整条系列成立，逐点虚检是受罚的。
- ⚠️ 依赖图例/列名匹配（NLS>0.5），无图例场景会丢系列；θ=1% 语义是"<1% 按比例、≥1% 全错"，比"RMSE≤1% 达标/不达标"的硬二值化更细腻；**建议真实图验收阶段启用**。

#### A5. ChartRM（ChartZero 2026）—— 端到端门控指标（可选项）

**精确定义（一手全文核实）**：`ChartRM = Φ_axis × (1/N) Σᵢ Ψ_legend,i · Γ_data,i`
- Φ_axis ∈ {0,1}：轴标签判错整图归零（门控）；
- Ψ_legend,i ∈ {0,1}：第 i 条曲线图例配对对错（门控）；
- Γ_data,i = exp(−λ · RMSE/Range(Y))：数据精度（连续）。
- 报数：ChartZero 全流程 0.921，传统管线 <0.07（"仅曲线追踪好不够"的实证）。

**点评**：把"轴→图例→数据"三级可用性串成一个 0-1 分，理念先进；但依赖图例配对/轴元数据（本项目 Phase 2 才完整），**建议真实图阶段再启用**。

#### A6. 图表理解基准（VQA 域）—— 间接参考，不可直接用于曲线提取评估

- **ChartQA**（2022）：**relaxed accuracy**（数值答对条件 `|pred−gt|/max(|gt|,1e-10) ≤ 5%`（部分论文用 1%），源自 PlotQA/Methani et al. 2020；非数值答案要求精确匹配——lm-evaluation-harness 官方实现核实）；官方另提供**数据提取评估脚本**：把 GT/预测表所有数值平铺，`distance=min(1, |x₁−x₂|/(|x₁|+1e-15))` + Hungarian + `score=1−cost/max(len)`——**点集匹配 + 值相对误差**，与 CHART-Info 同族（一手代码核实）。
- **PlotQA / MatCha / DePlot / UniChart / ChartX / CharXiv** 等：以 VQA accuracy / TEDS（表结构相似度）为主，**不直接度量曲线几何精度**，仅作背景（详见 §二 子代理 C 节；注："OmniChart（2024）"未查到对应论文，见 C-7 勘误）。
- **RMS（Relative Mapping Similarity，DePlot arXiv:2212.10505）**：把表格看作 (行头, 列头, 值) 三元组无序集合，键距离=归一化 Levenshtein（超阈值 τ 截 1），值距离 `D_θ = min(1, |p−t|/|t|)`（超 θ 截 1），最小代价匹配后 RMS_precision/recall/F1（τ/θ 取值 DePlot 正文未给出）；**RNSS**（ChartQA 提出，DePlot 形式化）：把表格看成**无序数值多重集**，`D(p,t)=min(1,|p−t|/|t|)`，Hungarian 后 `RNSS = 1 − ΣX·D/max(N,M)`。EpiCurveBench 实证两者局限：把"正确但平移 1 天"的提取判为近失败（key-value 范式无视时序结构）。**对本项目**：RNSS 丢失曲线归属与 x 信息，不可单独用；RMS-F1 可在"图例名↔数据值"关联评估中作补充。

### B. 像素级指标（分割/掩码域）—— 训练与诊断 ⭐

| 指标 | 定义 | 出处 | 本项目用途 |
|---|---|---|---|
| **IoU / mIoU** | 实例/类别掩码交并比 | ChartZero 报实例级 IoU 0.75→0.82（GOI 消融） | 训练 val_iou（已有，0.7938）与分割诊断 |
| **clDice** | 骨架 Dice（拓扑感知，细长结构专治） | arXiv:2003.07311（医学分割） | 细线断裂/连通性诊断（可选） |
| **6a 像素版** | CHART-Info 连续指标跑像素域 | metric6a.py（`compare_continuous` 直接用于 6a） | 与文献 6a 直接对比 |
| 像素误差直方图 | pred 点与 GT 掩码最小距离分布（px） | 本项目 S1 诊断（mean 0.56px / 偏移 +0.67px） | 亚像素精度监控 |

**点评**：像素级指标与"数据坐标精度"脱节（本项目已三次实证：像素指标升但端到端 6a 降）→ **只能做诊断，不能做主验收**。

### C. 序列/几何相似度指标（可选，真实图阶段）

- **DTW / ERP**：容忍 x 错位；ERP 有 gap 罚（截断惩罚）——ECS 即其变体（见 A4）；裸 DTW 不罚虚检、需先匹配；
- **Hausdorff / HD95 / Chamfer**：几何偏差辅助诊断（HD95 抗离群）；不罚虚检，不宜作主指标；
- **车道线协议（TuSimple accuracy/FP/FN、CULane 像素 IoU + F1）**：与"多曲线接近区归属 + 虚检/漏检"问题几乎同构，是**点→曲线距离阈值 + P/R/F1** 家族的最强参考（详见 §二 子代理 D 节）；

### D. 计数与结构指标

- **曲线数准确率**（N_pred = N_gt）：本项目现有（val 174/180）；对应 PseCo（CVPR'24）"分割+计数联合一致性"思想；
- **6b 的虚检惩罚**即为结构级指标（LineFormer 论文明确：6b 高分 ⟺ 精度高）。

---

## 二、子代理补充文献（A/B/C/D 已全部回填）

### 子代理 A（折线图提取评估：ChartDETR / CHART-UB 竞赛 / LineEX / DeepChart 等）✅已回填

**A-1. LineFormer 评估口径（代码级澄清，最重要）**
- 数据集：AdobeSynth19、UB-PMC22（train ~1500/test 158）、LineEX（40K/10K 子集）；**无 "SynthLine" 数据集**（项目文档若引用需更正）；
- **官方 6b（metric6b.py）在数据单位 'data series' 上计算并含图例名匹配（β=2）；LineFormer 仓库 eval.py 对 6a/6b 都在像素空间 'visual elements' 上计算（无图例名匹配）**——论文报告的 6b（UB-PMC 88.25）与竞赛官方口径不完全等价；本项目用官方数据值口径（见 §A1）；
- 骨架消融（UB-PMC 6a/6b）：Swin-T 93.1/88.25、ResNet50 93.17/87.21、Swin-S 93.19/88.51、ResNet101 93.1/90.08。

**A-2. ICDAR'19 CHART-Info / ICPR'20 CHART-UB 竞赛（官方 6a/6b 出处）**
- 竞赛官网：https://chartinfo.github.io/（2019/2020 任务页 + 官方指标文档 metrics/metric.pdf）；
- 三任务指标：文本检测=每块 IoU 和 ÷ max(#pred,#GT) + 归一化 CER；轴解析=修改版 F-measure（tick 按像素距离线性打分）；6a 元素（点/条/箱）=`max(0,1−(D/T)²)`，T=最短图像边长（⚠️ **版本差异**：metric.pdf 写 1%、官方代码 metric6a.py 与 README 用 5% 线性——以官方代码 5% 为准；折线元素用 6b 连续指标的像素版）；
- 2019 竞赛论文（Davila et al.）：https://ieeexplore.ieee.org/abstract/document/8978105

**A-3. OKS 关键点指标（LINEEX WACV'23 → ChartDETR）**
- **OKS(pᵢ) = exp(−dᵢ²/(2s²k²))**，dᵢ=预测点到最近 GT 点欧氏距离，s=图像对角线；sim_str：k=0.025、阈值 0.5、一对一贪心匹配；sim_rel：未过阈值的点若到 GT 线段垂距 < β=0.007·s 仍判 TP；
- ChartDETR（arXiv:2308.07743）报数：AS OKS_str F1 0.98（前最佳 0.71）、LE 0.95、EC 0.86；
- 点评：**像素关键点几何指标，不度量数据值误差、对 log 轴无感知**——仅作背景，不可用于本项目验收。

**A-4. ChartOCR（WACV'21，6a/6b 家族直接前身）**
- 折线指标与 CHART-Info 同源：区间加权相对误差 F1（原文公式**无 ε 项**，与官方实现 ε=1% 略有差异），多线用枚举组合匹配（非 Hungarian）；
- 报数：ExcelChart400K Line 0.962；FQA Mean Error Line 0.484（相对误差均值性质，与 RMSE 口径不同）；
- ChartReader（ICCV'23）直接沿用 ChartOCR 指标 + ChartQA ±5% 放宽准确率。

**A-5. "DeepChart KDD 2019"：未找到（更正项目文档引用）**
- DBLP KDD 2019 检索 chart 相关论文 0 匹配；同名 DeepChart（Signal Processing 2016）是图表**分类**论文（指标=分类准确率）；KDD 时代最接近的图表数据提取工作 = Liu & Klabjan（arXiv:1906.11906，§A2 已并入）与 DeepRule（arXiv:1812.01796）。
- → 项目 `材料曲线提取项目_后续工作论证报告.md` 等文档中若引用 "DeepChart KDD 2019" 需更正或删除。

### 子代理 B（曲线数字化评估：EpiCurveBench v1 ERP/NLD、CurveDE、WebPlotDigitizer 信度、ChartRecover 细节）✅已回填

**B-1. EpiCurveBench 虚检/漏检行为精确版（重要修正）**
- 整条虚检系列（NLS 匹配不上任何 GT）**忽略不计**；整条漏检系列**记 0 分**；
- 但**单点虚检**在 ERP 中走"插入"分支（每点罚 λ=1，计入 C 与分母 #gaps）、**单点漏检**走"删除"分支——**"虚检不罚"仅对整条系列成立**（项目 C_RESEARCH.md 的表述需按此修正）；
- 报告数值：Gemini 2.5 Pro 52.3%、GPT-5.2 43.9%、Claude 42.0%、Qwen3-VL 27.5%、TinyChart 11.4%、OneChart 9.4%；双标注员 ECS 90.9%（中位 95.4%）。

**B-2. Graph-FINDER nRMSE 精确定义**
- 原文图注："nRMSE is reported as a normalized pointwise-error metric **relative to the reference y-axis dynamic range**"；mean nRMSE=0.018 ≈ **1.8% 满量程**（可作为本项目 1% 验收线的外部参照基线）；mean Spearman ρ>0.92；
- 配对方式：**把参考曲线插值到提取 x 位置**再逐点计算——对 x 方向误差不敏感（x 误差转嫁为插值误差），x 轴误差需单独监控（Turner 2023 同结论）。

**B-3. ChartRecover（Comms Eng 2026）补充细节**
- Metric6a（像素域）：散点=欧氏距离/图像最短边（γ=β=1）；Metric6b（数据域）：散点=**马氏距离**（真值协方差归一）、条形=曼哈顿（η=min(σ,|μ|/20)）；匹配=匈牙利；
- **相对误差在数值接近 0 时急剧放大**（"increases sharply as coordinate approaches zero"）——log 轴低值端尤甚，支持"log 域单独处理"建议；
- 报数：Metric6b 散点 0.867/0.898（UB-PMC22/CHART-Info24）；10% 阈值准确率 >80%、2% 阈值条形图 >70%。

**B-4. 人工数字化金标准上限（验收标尺）**
- Drevon et al. 2017（Behavior Modification）：2 名编码者、36 图、168 序列、3,596 点，高信度效度（具体 ICC 付费墙未公开）；
- Turner et al. 2023（RSM, PMC 开放）：**x 轴时间点误差 9%-35%**（刻度对齐是主因）、**y 轴提取非常准**；Burda：ICC>0.95、与原数据差异 0.3%-8.92%；van der Mierden：CCC>0.99（结果）/0.92（SE）；
- **结论**：1% 满量程验收线处于人工一致性区间内（合理但偏严）；x 轴/刻度对齐误差必须在评估中单独监控。

**B-5. 图转表指标谱系（综述 arXiv:2410.13883）**
- **RNSS**（ChartQA）：仅数值集合相似度；**RMS**（DePlot）：整表映射级相似度；**SCRM**（StructChart）：图表反渲染结构保真（STR 三元组）；**RD**（SIMPLOT）：行列表映射的最小代价匹配——均为"表/结构"域指标，与曲线几何精度正交。

**B-6. CurveDE（材料领域直接先验）**
- "CurveDE: An image recognition tool for extracting stress-strain curve data of Superalloys"，Computational Materials Science，DOI 10.1016/j.commatsci.2025.114430（Semantic Scholar API 核实）；**全文付费墙，评估方式未查到**（子代理 B 与本人均确认），引用时标注"按原文核查"。

**B-7. EpiCurveBench v1（medRxiv 2025.09.23.25336494）→ v2/arXiv 2605.27195**
- 项目 C_RESEARCH.md 记录的 v1 口径（NLD>0.5 图例配对 + ERP：gap 惩罚、<1% 量程按比例罚、≥1% 记全错；漏检记零）与 v2 的 ECS 设计一致（NLS 阈值 0.5、θ=0.01；"NLD"在论文中实为 NLS=Normalized Levenshtein Similarity）；**最终以 v2 ECS 为准**（见 §A4 与 B-1 的虚检行为修正）。

### 子代理 C（图表理解基准：PlotQA/MatCha/UniChart/OmniChart/ChartX/CharXiv/ChartBench 的提取相关指标）✅已回填

**C-1. PlotQA（ICCV'19, arXiv:1909.00997）**：数据/推理 QA 基准；**relaxed accuracy 即 PlotQA 首创**（数值答对容差 |pred−gt|/|gt| ≤ 5%，ChartQA 沿用）；其 **VED（Visual Element Detection）模块用 mAP@IoU（0.5/0.75/0.90）** 评估元素检测（报告 ~94.92% mAP@IoU0.90）——**检测级 mAP 可借鉴**（判"曲线/元素是否检出"），但与数值误差互补而非替代。

**C-2. MatCha（ICML'23/ACL'23, arXiv:2212.09662）**：chart derendering 仅为预训练目标，**下游只有 ChartQA/PlotQA 的 VQA 评测**（relaxed correctness 5% 容差 + Chart-to-Text BLEU4，一手全文核实）——**无独立图转表评测基准**，无曲线几何指标。**TEDS 出处澄清**：TEDS（树编辑距离相似度）来自表格结构识别 **PubTabNet**（Zhong et al., ECCV'20, arXiv:1911.10683），`TEDS = 1 − TED/ max(|T_a|,|T_b|)`——属表格结构指标，非曲线提取指标。

**C-3. DePlot（ICLR'23, arXiv:2212.10505）**：**RMS（Row-Match Similarity）** = 预测表与 GT 表行级最优匹配下的 `1 − ΣX_ij·D_ij/ΣX_ij`（D_ij 为行距离；公式细节需查原文 §4）；图转表任务最接近"提取质量"的度量，曲线项目可把"（曲线名,x,y）"组织成三元组集合套用（详见 §A6/D6）。

**C-4. UniChart（EMNLP'23）/ ChartGemma（2024）/ ChartX（2024）/ CharXiv（NeurIPS'24）/ ChartMimic（2024）**：
- UniChart：5 任务（ChartQA/摘要/OpenCQA/事实核查/图转文本）评测，图转表用 RNSS + RMS-F1；无曲线级数值误差度量；
- ChartGemma：指令微调图表推理，沿用各基准官方指标（ChartQA relaxed accuracy 等）；
- ChartX（arXiv:2402.12185）：22 任务 × 18 图表类型基准，任务级 accuracy/F1，**无曲线级 MAE/RMSE 式度量**；
- CharXiv（arXiv:2406.18521）：2,323 张 arXiv 真实图，描述性/推理性 QA——问答类，无提取度量；
- ChartMimic（arXiv:2406.09961）：图→代码，代码级+图像级两级评估（代码执行 + 渲染比对），间接验证数值但非逐点误差指标。
- 与 EpiCurveBench 结论互相印证：**VLM 系在真实曲线数字化上精度不可靠（最强仅 52.3% ECS）**。

**C-5. 元素检测基准（mAP@IoU 家族）**：PlotQA VED（mAP@IoU0.9≈94.92%）、Ganguly et al.（AAAI'21, arXiv:2007.02240，PlotQA 元素检测 mAP 评测范式）、ChartDete（ICDAR'23, arXiv:2305.04151，上下文感知元素检测）、LineEX（WACV'23，OKS 关键点，见 A-3）——均为"检测有没有"层面，与 6a/6b 的"提取准不准"互补。

**C-6. 2024-2026 新基准补充**：
- ChartBench（arXiv:2312.15915）：复杂推理 QA，规则/正则匹配判正确性；
- ChartInsights（EMNLP'24 Findings, arXiv:2405.07001）：低层图表 QA，任务含 data extraction/visual comparison（指标细节未核实）；
- ChartQAPro（2025, arXiv:2504.05506）：24 类图表 QA，沿用 relaxed accuracy 系列；
- Self-Ensembling VLM for Chart Data Extraction（2026, arXiv:2605.27298）：图→表提取用 **RMS-F1（三元组集合 F1）**；
- Y 轴偏置研究（2026, arXiv:2604.24987）：图转表评测用 **RNSS + SES（Swapping Error Score）**。

**C-7. 名称勘误（供项目文档修正）**：
- **"OmniChart（2024）"未查到**——最接近的是 Omni-Chart-600K（NAACL'25 Findings 数据集论文）；
- **"CEBench"、"ChartPixel"、"ChartElements" 未查到**图表领域对应物（CEBench 同名是 LLM 成本基准、ChartPixel 是商业工具）；
- "DeepChart KDD 2019" 未找到（见 A-5）。

**C-8. 结论**：图表理解基准线对本项目**只有背景/对齐价值**（证明"通用 VQA 指标测不了曲线提取质量"）；直接可复用的只有 ①LineFormer 最优匹配+平均相似度（6a/6b 族）、②RMS/RMS-F1（三元组/行匹配）、③检测级 mAP@IoU、④relaxed accuracy（粗粒度判对率）——主验收仍回到 §A1-A4 曲线级指标族。

### 子代理 D（经典曲线相似度：DTW/Hausdorff/Chamfer、车道线 TuSimple/CULane、线段检测、clDice/SRL、统计指标）✅已回填

**D1. DTW（动态时间规整）**
- 定义：`D(i,j)=d(qᵢ,cⱼ)+min{D(i−1,j), D(i,j−1), D(i−1,j−1)}`，容忍 x 轴非线性伸缩；归一化方式有 z 归一化（序列级）或除以对齐路径长度 L（分数级）。
- 代表出处：时间序列分类标准基线（arXiv:1401.3973）；EpiCurveBench 用 ECS 对比 DTW（arXiv:2605.27195）；ERP 为其"编辑距离 + 实数惩罚"变体（Chen & Ng, VLDB 2004）。
- 点评：✅容忍采样不均/轻微错位；❌无 gap 罚（截断可蒙混）、不罚虚检、需先做曲线匹配、对交叉错配不敏感。

**D2. Hausdorff / HD95 / Chamfer**
- 双向 Hausdorff `H(A,B)=max{h(A,B),h(B,A)}`；医学分割常用 **HD95**（95 分位，抗离群）；Chamfer = 互为最近点距离的平均（点云标配）。
- 点评：Hausdorff 对单点毛刺极敏感、不罚虚检；Chamfer 平均化稀释局部误差——**只适合做辅助诊断，不适合主指标**。

**D3. 车道线检测协议（与"接近区归属"问题同构，强参考）**
- **TuSimple**：accuracy（点级正确率，预测点距 GT 点 <阈值 判正确）+ **FP=F_pred/N_pred**（虚检率，分母=预测数）+ FN——与"多曲线虚检"评估几乎同构；
- **CULane**：F1-measure，预测/GT 车道线渲染为宽掩膜后按像素 **IoU** 匹配判 TP（区域式匹配，天然处理 x 采样不均）；
- 出处：TuSimple 官方基准（gitcode tusimple-benchmark doc/lane_detection）、CULane（Pan et al., CVPR'18）、CLRNet（arXiv:2203.10350）。

**D4. 线段/曲线检测评估**
- Wireframe/YorkUrban 报 **junction-mAP、线段 P/R、AP⁵（5px 容差下的平均精度）**（Deep Hough-Transform Line Priors, ECCV'20；HAWPv2）；
- 曲线/边缘 one-to-one 成本匹配显式用于评估：SCITEPRESS 2020（DOI 10.5220/0009330005900598）。

**D5. 分割域拓扑指标**
- **clDice**：`Tprec=|S_P∩V_L|/|S_P|`（预测骨架落入 GT 体素比例）、`Tsens=|S_L∩V_P|/|S_L|`、`clDice=2·Tprec·Tsens/(Tprec+Tsens)`（arXiv:2003.07311）；
- **SRL**（ECCV'24, arXiv:2404.03010）：只在 GT 骨架像素上算 BCE，保护细长结构连通性（注意存在质疑论文 arXiv:2508.11374）；
- 点评：拓扑/连通性诊断专用，与车道线/线段协议互补。

**D6. 图转表域的 RMS（Relative Mapping Similarity）**
- DePlot（arXiv:2212.10505）：对预测表与 GT 表做映射级 P/R/F1（RMS_recall = 1 − ΣΣ XᵢⱼD(·)），后被图表提取工作广泛采用（arXiv:2605.27298 等）；EpiCurveBench 明确批评 RMS/SCRM 把"正确但平移 1 天"判为失败（key-value 范式局限）。

**D7. 统计指标在数字化验证中的用法**
- WebPlotDigitizer 信度/效度验证（Drevon et al., SAGE 10.1177/0145445516673998）；中断时间序列图数字化研究（Wiley 10.1002/jrsm.1646，发现 **x 轴提取精度系统性差于 y 轴**——与本项目刻度 OCR 瓶颈一致）；自动化生存曲线数字化（BMC 10.1186/s12874-024-02273-8）。

**子代理 D 总结建议（与主报告结论互相印证）**：①曲线级 one-to-one 匹配 + 点→曲线距离阈值 P/R/F1（TuSimple 式，含虚检惩罚）；②匹配后逐点 nRMSE/R²（数值精度）；③clDice/SRL 或 HD95 做拓扑补充——与主报告"6a/6b + nRMSE + 像素级诊断"三件套同构。

---

## 三、关键辨析：本项目现状与文献口径的 7 处差异

（对照文件：`scripts/eval_multi_6ab.py`、`src/mci/eval/metrics.py`、官方 `metric6a.py`）

| # | 维度 | 现状（本项目） | 文献主流（官方） | 影响与建议 |
|---|---|---|---|---|
| 1 | **匹配算法** | 贪心（每轮取全局最小代价对） | **Hungarian**（linear_sum_assignment 最大化总相似度） | 接近区多曲线时贪心可错配；改 Hungarian 一行代码，原则性正确 |
| 2 | **归一化分母** | GT 曲线 y 跨度 | 验收口径=**轴满量程**（ChartZero 同）；**Graph-FINDER nRMSE 与 ECS 用"曲线/参考 y 动态范围"（跨度族）**；官方逐点误差=值相对 | ⚠️ **实证（val 540 条，chain 模型）**：跨度版 rel 中位是满量程版的 **1.65×**（fs/span 中位 0.605）；1% 阈值下 span 版召回 459/540=85.0%，满量程版 **478/540=88.5%**，翻转 19 条全部是"span 失败→fullscale 通过"（反向 0）——**平缓曲线（跨度≪轴范围）用跨度归一更严格**。两口径都有权威背书（跨度=ECS/Graph-FINDER；满量程=验收标准/ChartZero），**验收标准明确=轴满量程 → 建议双报**（span 版与 ECS/Graph-FINDER 可比，fullscale 版与验收对齐） |
| 3 | **误差聚合** | RMSE（L2，重叠区内） | 官方=逐点相对误差按 x 区间加权均值 | RMSE 对离群点敏感但稳定；官方值相对误差对 log 轴小值端爆炸——本项目 log 轴多，**保留 RMSE 族为主**，官方 F1 版为辅 |
| 4 | **成功判据** | 二值化（rel≤1%） | 官方=连续相似度（0-1）；ChartRecover=多阈值 Accuracy(τ) | 二值化直观且对接验收；建议**同时报连续 6a/6b**（与文献可比） |
| 5 | **x 覆盖** | 只统计重叠区（截断可蒙混） | 官方：GT 点全参与（插值钳位→截断必受罚） | 加覆盖项：未覆盖 GT 段按满误差计，或 rel-RMSE × (1−uncovered_frac) 修正 |
| 6 | **聚合** | 全数据集池化（曲线等权） | 逐图打分再平均（图等权） | N_gt 不均时两法不同；官方口径=逐图平均；建议报告两版 |
| 7 | **虚检** | 6b 分母 max(Ng,Np)（已对齐） | 同（LineFormer K 公式） | ✅ 已一致，无需改 |

**另注意**：验收标准"多曲线召回率 ≥95%"未定义"召回"的精确定义——本报告建议以 **6a（Hungarian + 满量程归一 + rel-RMSE≤1%）** 为唯一权威口径并在验收报告中写死公式。

---

## 四、权威指标推荐（最终结论，含精确定义）

### 4.1 主指标集（验收/竞赛/论文全部使用）

**指标 1：6a 召回率（纯召回）**
```
配对：cost(i,j) = RMSE(G_i, P_j)  # P_j 在 G_i 的 x 网格上插值求 y 误差（全 x 域，含未覆盖段惩罚）
      用 Hungarian 最大化匹配（最小化总 RMSE 代价，未匹配 GT 记 ∞）
成功：rel_i = RMSE(G_i, P_match) / Y_range(轴满量程) ≤ 0.01
6a = Σᵢ [rel_i ≤ 0.01] / N_gt          # 按图计算后对图平均（官方口径），同时报池化版
```

**指标 2：6b 召回率（罚虚检）**
```
同 6a，但 6b = Σᵢ [rel_i ≤ 0.01] / max(N_gt, N_pred)
（未匹配的预测曲线记 0 分——LineFormer K 公式）
```

**指标 3：nRMSE 分布（数值精度）**
```
nRMSE_i = RMSE(G_i, P_match) / Y_range   # Y_range = meta y_range 满量程（验收口径）
报告：中位 / p90 / 最大 / ≤1% 达标率（pass_rate）/ ≤1% 曲线数
桶分布：≤1% / 1-2% / 2-5% / >5%（>5% 需逐条归因）
```

### 4.2 辅助/诊断指标

- **曲线数准确率**：N_pred = N_gt 的图占比（已有）；
- **官方连续 6a/6b**（CHART-Info F1 版）：与 LineFormer/UB-PMC 数值可比（新增脚本）；
- **分层报告**：按轴型（lin/log）、模板、曲线形态、退化类型分层报 6a/nRMSE（定位失败模式）；
- **像素级**：训练 val_iou（已有）+ 像素误差直方图（S1 已有方法）。

### 4.3 可选项（真实图阶段/科研加分）

- **ECS**（EpiCurveBench）：真实图"可用性"主指标，配 NLS 图例配对；其"整条漏检记 0、单点虚检/漏检按 gap 罚"的行为已精确化（见 §二 B-1）；
- **ChartRM**（ChartZero）：全链路门控分（轴+图例+数据）；
- **Spearman ρ**（Graph-FINDER）：形状一致性（材料论文加分项，与 nRMSE 双报）；
- **标注一致性**：双人标注 gold 时用 ECS 或 ICC/CCC 报标注者间一致性（EpiCurveBench 90.9%；Burda ICC>0.95、van der Mierden CCC>0.99——**人工金标准上限 ≈0.95-0.99 一致性、数值差异 0.3%-9%**，1% 满量程验收线位于该区间内，合理但偏严）；
- **log 轴专门处理**：主流指标均无 log 轴标准（ChartZero 基准 388/1000 张含 log 轴也只做分层）；建议 ①log 域计算 nRMSE（分母=log 量程）或相对误差口径；②报 ChartRecover 式 2%/5%/10% 多阈值准确率（ε_v=|v_pred−v_gt|/(|v_gt|+1e-8)）；③**按 lin/log 轴分层报告**（ChartZero 基准先例）。

### 4.4 与验收标准映射表

| Prompt.md 验收 | 权威指标 | 目标值 |
|---|---|---|
| 曲线数据点 RMSE ≤ 轴满量程 1% | nRMSE ≤ 0.01（中位/达标率），分母=轴满量程 | 合成集 100% ≤1%（单曲线已 99.6%）；多曲线 ≥95% |
| 多曲线召回率 ≥95% | **6a**（Hungarian + 满量程归一 + rel≤1%） | ≥0.95（当前 span 口径 0.8481；**满量程口径实证 0.885**，差距 ~6.5pp，见 §三.2） |
| 坐标类型判断 ≥98% | 轴型判型准确率（已有 fit_axis 统计） | ≥0.98（已达标） |
| 单图 <5s GPU | 耗时（冷启动/稳态双口径） | <5s |

---

## 五、实施建议（代码落地清单）

1. **升级 `scripts/eval_multi_6ab.py`**：
   - 贪心 → `scipy.optimize.linear_sum_assignment`（代价 = RMSE，含"未覆盖惩罚"版本）；
   - **确认在数据值坐标空间评估**（官方 6b 口径；不要学 LineFormer 仓库的像素空间做法——见 §A1 澄清）；
   - 新增 `--norm {span, fullscale}`：span=现状（曲线跨度），fullscale=验收（meta y_range，缺 meta 时退回跨度并标注）；
   - 新增"连续 6a/6b"输出（官方 metric6a.py 移植为 `src/mci/eval/chartinfo_metrics.py`，含逐点相对误差 + ε=范围/100 + x 区间加权 + F1 + Hungarian + 可选图例名匹配 β=2）；
   - 聚合双报：逐图平均 + 池化。
2. **`src/mci/eval/metrics.py`**：`curve_metrics` 增加 fullscale 归一参数与 x 覆盖惩罚项（`rel_rmse_fs`、`x_coverage` 已有）；
3. **新增 `scripts/eval_ecs.py`**（可选，真实图阶段）：NLS 配对 + ERP + θ=0.01（递推公式见 §A4，可直接复现）；
4. **tests**：为 Hungarian 匹配、满量程归一、连续 6a/6b 补单元测试（现 `test_metrics.py` 覆盖 RMSE/覆盖）；
5. **log 轴分层**：所有指标按 lin/log 轴分层报告（ChartZero 基准先例），log 图另报 log 域 nRMSE 与 2%/5%/10% 多阈值准确率；
6. **验收报告结构**（PHASE_D_ACCEPTANCE.md §五 对齐）：6a/6b + nRMSE 分布 + 曲线数 + 分层 + 失败桶归因 + 与文献数值对比表（LineFormer UB-PMC 93.1/88.25、ChartZero 0.028 NRMSE、Graph-FINDER 0.018 nRMSE ≈ 1.8% 满量程、ECS 52.3% SOTA）。
7. **§三.2 实证分析脚本**：`data/_analyze_norm.py`（读 `data/eval_multi_diag_chain/diag.json` + val 集 meta.json，重算跨度/满量程双口径召回与翻转数），换模型后可直接复跑。

---

## 六、参考文献（一手来源）

1. CHART-Info 竞赛官方指标定义与任务说明：https://github.com/adobe-research/CHART-Synthetic （`metric6b.tex`、README Tasks 6a/6b、`metric6a.py`/`metric6b.py`）；官方指标文档 https://chartinfo.github.io/metrics/metric.pdf ；2019 竞赛论文 https://ieeexplore.ieee.org/abstract/document/8978105
2. LineFormer 论文全文（§7 Evaluation）：https://ar5iv.labs.arxiv.org/html/2305.01837 ；官方代码（含 CHART-Info metric6a.py）：https://github.com/TheJaeLal/LineFormer
3. EpiCurveBench 全文（ECS/NLS/ERP 定义）：https://arxiv.org/abs/2605.27195
4. ChartZero 全文（IoU/NRMSE/ChartRM）：https://arxiv.org/abs/2605.05820
5. ChartRecover（Nature Comm. Eng. 2026，Metric6a/6b + 多阈值 Accuracy(τ)）：https://www.nature.com/articles/s44172-026-00691-8
6. Graph-FINDER（npj Comput. Mater. 2026，Spearman ρ + nRMSE）：https://www.nature.com/articles/s41524-026-02247-y
7. ChartQA 官方数据提取评估代码：https://github.com/vis-nlp/ChartQA （`Data Extraction/evaluate_data_extraction.py`）；relaxed accuracy 实现：lm-evaluation-harness `lm_eval/tasks/chartqa/utils.py`
8. FigureSeer（ECCV 2016，2% 点阈值）：https://link.springer.com/chapter/10.1007/978-3-319-46478-7_41 ；Linear Programming（WACV 2022）：https://www.openaccess.thecvf.com/content/WACV2022/html/Kato_Parsing_Line_Chart_Images_Using_Linear_Programming_WACV_2022_paper.html
9. Liu & Klabjan（1%/5%/10%/25% 相对误差阈值 + ALL 判据）：https://arxiv.org/abs/1906.11906
10. LINEEX（WACV'23，OKS_str/OKS_rel 出处）：https://www.openaccess.thecvf.com/content/WACV2023/html/P._LineEX_Data_Extraction_From_Scientific_Line_Charts_WACV_2023_paper.html ；ChartDETR：https://ar5iv.labs.arxiv.org/html/2308.07743 ；ChartOCR：https://openaccess.thecvf.com/content/WACV2021/papers/Luo_ChartOCR_Data_Extraction_From_Charts_Images_via_a_Deep_Hybrid_WACV_2021_paper.pdf ；ChartReader（ICCV'23）：https://ar5iv.labs.arxiv.org/html/2304.02173
11. DePlot（RMS/RNSS 定义）：https://ar5iv.labs.arxiv.org/html/2212.10505 ；UniChart：https://aclanthology.org/2023.emnlp-main.906.pdf ；图表理解综述（RNSS/RMS/SCRM/RD 谱系）：https://ar5iv.labs.arxiv.org/html/2410.13883 ；PlotQA：https://arxiv.org/abs/1909.00997 ；MatCha：https://ar5iv.labs.arxiv.org/html/2212.09662 ；ChartX：https://arxiv.org/abs/2402.12185 ；CharXiv：https://arxiv.org/abs/2406.18521 ；ChartMimic：https://arxiv.org/abs/2406.09961 ；TEDS 出处 PubTabNet（ECCV'20）：https://arxiv.org/abs/1911.10683 ；ChartInsights：https://arxiv.org/abs/2405.07001 ；ChartQAPro：https://arxiv.org/abs/2504.05506 ；图转表自集成 VLM（RMS-F1）：https://arxiv.org/abs/2605.27298 ；Y 轴偏置（RNSS+SES）：https://arxiv.org/abs/2604.24987 ；元素检测：Ganguly et al.（AAAI'21, arXiv:2007.02240）、ChartDete（ICDAR'23, arXiv:2305.04151）
12. WebPlotDigitizer 信度（Drevon 2017）：https://pubmed.ncbi.nlm.nih.gov/27760807/ ；Turner 2023（x 轴误差主源）：https://pmc.ncbi.nlm.nih.gov/articles/PMC10946754/ ；clDice：https://arxiv.org/abs/2003.07311 ；SRL（ECCV'24）：https://arxiv.org/abs/2404.03010
13. 本项目现有：`scripts/eval_multi_6ab.py`、`src/mci/eval/metrics.py`、`PHASE_D_ACCEPTANCE.md`、`NEXT_STEPS_PROMPT.md`
14. 子代理完整报告存档：`F:\CODE\New\linechart_extraction_metrics_research.md`（A）、`F:\CODE\New\curve_metrics_research\曲线提取评估指标调研报告.md`（B）
