# Phase C 深入研究与方案选择（2026-08-19 三路并行调研汇总）

> 调研方式：3 个并行研究子代理 × (web_search + 论文/官方文档抓取)，
> 核实 25+ 来源；另加本会话本地失败模式诊断（180 图逐曲线）。

## 一、文献地图（2023-2026 图表数据提取）

### 1.1 实例分割范式（与本项目同源）
- **LineFormer**（ICDAR'23, arXiv:2305.01837）：Mask2Former 式实例分割。
  关键设计：**交叉处允许像素多标签**（Bresenham 3px 重叠 GT 掩码）；
  评估用 **6a/6b 双口径**（6a 只罚漏检；6b 罚虚检）。UB-PMC（真实论文图）
  6a/6b = 93.1/88.25。社区微调案例：预训练模型**过度检测 +62.3%**，域内
  微调后 -52.7pct → 虚检是召回被拖低的头号元凶。
- **ChartZero**（arXiv:2605.05820, 2026）：纯合成 10 万图零样本。
  双头 U-Net（语义头+实例嵌入头）+ **CoordConv** + GroupNorm + **GOI 损失**：
  类内拉近（余弦）+ 类间全局正交（质心点积平方和）+ **小簇合并**
  （交叉碎片并入最近大实例，IoU 0.75→0.82, NRMSE 0.071→0.028）。
  已知失败：急角度交叉互换 4.2%、密集簇幻影曲线 3.9%。
- **Graphical Models 2025**（S1524070325000062）：Mamba 增强 Transformer +
  **GT 掩码引导的中间层训练**（深度监督，+10%）；PMC 真实集转换精度 87.91%。
- **AI-ChartParser**（CGF 2025）：多任务（元素+拐点+曲线）+ 区间-均值映射。
- **Graph-FINDER**（npj Comput. Mater. 2026）：免校准多线数字化，nRMSE 0.018；
  用 **Spearman ρ + nRMSE 双指标**（形状/数值解耦）。
- **EpiCurveBench**（medRxiv 2025）：100 真实图；**NLD>0.5 图例配对 + ERP
  容错序列距离（gap 惩罚、<1% 量程按比例罚、≥1% 记全错）**；虚检序列不罚、
  漏检记零——最干净的多曲线召回口径；最强 VLM 仅 42.9%（Advanced）。

### 1.2 细长结构分割与损失
- **clDice**（arXiv:2003.07311）：骨架感知 Dice，专治细长结构断线/拓扑错误；
  血管分割实证有效。**Skeleton Recall Loss**（ECCV 2024）为其升级版（开源）。
- **Focal Tversky**：小结构 +25.7%（医学分割消融）。
- **Boundary Loss**：边界定位误差。
- 白板笔画评价协议：标准 IoU 掩盖薄线失败 → 建议加薄线子集指标。

### 1.3 追踪/后处理
- **Steger 亚像素脊线**：±0.5px→±0.1px 精度（有成熟实现）。
- **骨架交叉点分支配对**：骨架度≥3 结点=交叉 → 切分支 + Hungarian 配对
  （方向夹角+通道置信度+图例色距离代价）→ Dijkstra 全局重建。
- LineFormer 后处理：x 区间定步长采样 + 线性插值补断口。
- WebPlotDigitizer 颜色分离（HSV 聚类图例区参考色）。

### 1.4 训练策略
- STU-Net：宽度+深度联合缩放持续提升（base 64→128 = 2.1M→8.4M 真实收益）。
- 分辨率是细线任务首要杠杆（SMILE-UHURA 血管分割、小目标综述）。
- AMP + EMA 零风险必做；copy-paste 免费多样性；域随机化多样性>数量。
- 通道顺序歧义：固定通道需规范排序（x 起点）或匈牙利匹配（NeurIPS'22）。

## 二、本地诊断结论（180 图逐曲线）

| 失败类 | 数量 | 根因 |
|--------|------|------|
| 1-2% 边缘 | 83 | log 轴 1px≈2% rel；模型概率峰值 +1~2px 系统偏移 |
| 2-5% | 47 | 训练目标错误（25.5% log 图 linear 拟合）+ 局部精度 |
| >5% (local_spike) | 41 | 交叉/靠近处通道归属混淆 + 追踪跳线 |
| 虚检（pred=5） | 19 图 | 空通道幻影（LineFormer 同类问题） |

**根因（已证实）**：训练侧 _fit_axis_from_labels 与评估侧 build_axes 轴型
判定不一致（y 轴 25.5%、x 轴 1.8% 训练图 log→linear 误判，log 轴上掩码
错位 100+px）→ 25% 训练目标系统性错误 → 训练平台化 + systematic_bias。

## 三、方案选择（按证据强度 × 成本）

| 方案 | 证据 | 成本 | 决策 |
|------|------|------|------|
| A. 训练目标修复（已实施） | 根因实证 | 已做 | **立即重训验证** |
| B. EMA + clDice/SRL 损失 | 细长结构强证据 | 低（代码 20 行） | 随 A 一并重训 |
| C. 推理侧：交叉分支配对+Hungarian | 明确对应 41 条 >5% | 中 | 重训后评估 |
| D. 虚检抑制（置信度/IoU-NMS） | LineFormer 社区实证 | 低 | 重训后评估 |
| E. GOI 损失（嵌入头+正交+merge） | ChartZero 消融强 | 高（改架构） | 观察 A+B 后决定 |
| F. CoordConv | ChartZero 消融 +0.06 IoU | 低 | 随 E 或独立验证 |
| G. 6a/6b 双口径评估 + 薄线子集 | LineFormer/EpiCurve | 低 | 立即（评估基建） |
| H. 曲线数语义先验（VLM） | PlotPick 88-96% | 中 | 真实图阶段 |

**推荐（本会话执行）**：A 已实施；G 立即补（评估口径）；
重训 = A+B（修复 + EMA + clDice），单一可控变量组合，预期 log 轴图
与 systematic_bias 类大幅回升；重训后按结果决定 C/D（后处理）与 E/F（架构）。

## 四、重训命令（修复版）

  $PY train/train_segmentation_multi.py --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single --val-dir data/val_multi,data/val_single --epochs 25 --batch 8 --size 512 --per-dir-limit 600 --init models/checkpoints/unet_multi_curve_512c.pt --out models/checkpoints/unet_multi_curve_evalfix.pt

训练后：$PY scripts/eval_multi.py --model models/checkpoints/unet_multi_curve_evalfix.pt --out-dir data/eval_multi_evalfix --size 512