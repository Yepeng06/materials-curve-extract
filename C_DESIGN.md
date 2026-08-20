# Phase C 设计：多曲线实例分割（U-Net K 通道）

> 状态：调研复核完成（2026-08-16，LineFormer arXiv:2305.01837 实例分割范式 +
> 现有 U-Net 架构扩展），待实施。范围：GPU 训练（RTX 4060 8GB）。
> 验收：多曲线召回 ≥95%；单曲线管线（unet_curve.pt + stub/paddle）不回退。

## 一、方案选择（调研依据）

| 方案 | 评价 |
|------|------|
| LineFormer 式实例分割（检测+回归） | 学界 SOTA（arXiv:2305.01837），但实现复杂、社区权重为电池曲线域 |
| **U-Net K 通道**（本方案） | 数据已就绪（train_platform 2000 张 + curves_px 像素点），现有 U-Net 架构直接扩展，8GB 显存可训；曲线数 ≤4（K=4） |
| 颜色分离 + top-K 组件 | 平台数据曲线为灰阶（实测 img_0000 6 种灰度），颜色分离不可靠；仅作推理后处理辅助 |

## 二、数据与实例掩码

- 训练：train_platform 2000 张（2-4 曲线）+ 新生成 600 张单曲线平台图（--num-curves 1，
  独立 seed）→ 覆盖 1-4 曲线；验证：新 seed 多曲线 120 + 单曲线 60
- **实例掩码生成**：meta.curves_px（每曲线像素点）→ cv2.polylines 连线（线宽=meta
  line_width 或 4px）→ K=4 通道掩码（0/1，通道按曲线 x 起点排序：最左→通道 0）
- 曲线数 <4：多余通道全 0

## 三、模型与训练

- UNet(in=1, base=64, out_channels=4)：仅改输出头（Conv2d(base, 4)）；
  初始化：加载 unet_curve.pt（encoder 权重共享，单曲线知识迁移）
- 损失：每通道 BCE + soft-Dice（复用 bce_dice_loss，多通道逐通道平均）
- 训练：size 512（与现有模型一致）、batch 8-16（8GB 显存）、40-60 epoch、
  Adam 1e-3 + Cosine、AMP；增强复用 _augment（注意：通道掩码同步增强）

## 四、推理与多曲线提取

- 4 通道 logits → sigmoid → 阈值 0.5 → 每通道独立骨架化+追踪（复用 curve_extractor）
  → 像素点 → 坐标映射（复用 build_axes）→ 曲线列表
- 通道即顺序（x 起点排序）→ 输出曲线按序排列；空通道（点数 <min）跳过
- 与 GT 匹配：预测曲线数与 GT 曲线数对比 → 召回（每条 GT 至少 1 条预测匹配）；
  逐曲线 RMSE（最近邻 x 对齐）

## 五、评估与回归

1. 新增多曲线评估：eval_multi.py（召回率/逐曲线 RMSE/F1）——独立 seed 生成
2. 单曲线回归：pytest + stub 合成/平台 + paddle 99% 不回退（默认 unet_curve.pt 不变；
   多曲线模型为独立 checkpoint，extractor 加 segmenter 参数 multi_unet）
3. 训练过程监控：val_iou（多通道平均）、召回随 epoch

## 五.5、实施记录

| 轮次 | 内容 | 结果 |
|------|------|------|
| v1 训练 | 2600 张（2-5 曲线），40 epoch，val_iou 0.606 | 召回 21.5%：曲线数不足（4 曲线图仅激活 3 通道） |
| v2 重训 | +600 张 4 曲线 +300 张 5 曲线 + 空通道损失降权，val_iou 0.635 | 召回 21.1%（曲线数恢复 106/180 但 RMSE 不达标）——**灰度聚类实例掩码不可靠**（虚线/抗锯齿灰度混乱，img_0200 聚类错乱） |
| v3 重训 | **curves_px 折线实例掩码**（位置精确），val_iou 0.632 | 召回 33.7%（曲线数 168/180）：位置偏差 2-11px——**256 分辨率精度上限** |
| 对照 | 单曲线模型（512）在 val_single：**100% 召回** | 确认 512 训练是精度关键 |
| 组合策略 | 单曲线语义（512 位置）+ 多曲线 argmax 归属 | 25.4%：归属在曲线靠近处不可靠 |
| 512 微调 | v3 → 512 微调 10+20 epoch（2100 张） | 召回 48.5%（位置精度改善，曲线数 94%） |
| v4（关键） | **CSV 重建实例掩码**：GT csv 160 点经刻度标签映射重建（与评估基准一致，验证 0.4px） | **召回 66.7%**（单曲线 91.7%、多曲线 64.6%，曲线数 92%） |

**结论（2026-08-18）**：多曲线 K 通道方案验证可行（召回 66.7%，单曲线 91.7%）。
三个关键决策：① 512 分辨率（256 是精度瓶颈）；② curves_px 折线→CSV 重建掩码
（训练目标必须与评估基准一致）；③ 灰度聚类否决。
**剩余差距**：多曲线归属（曲线靠近处通道混淆）+ 边缘精度；继续训练 +20 epoch
预计单曲线 →95%、多曲线 →70-75%；归属问题需进一步研究（可能用图例/颜色辅助）。

**根因突破（2026-08-19 会话）**：训练目标与评估基准并非真正一致——
train_segmentation_multi._fit_axis_from_labels 用「标签中心 polyfit +
`c[1] < img_h-85` 过滤」重建轴映射，与评估侧（detect_structure + read_ticks +
build_axes）**轴型判定不一致**：底部 y 刻度标签（如 0.001@527px）被 515px 过滤
排除后，剩余序列 0.1/1/10 比值=100 不触发 log 判定 → 判 linear；评估侧保留
底部标签 → 判 log。全数据核查：**y 轴判型不一致 904/3548 (25.5%)、x 轴 65/3680
（1.8%）**，全部为 fit=linear vs GT=log。log 轴上 linear 拟合误差可达 100+px，
模型被 25% 的错误目标训练 → 解释：① systematic_bias 类失败（同图 4 曲线同时
偏 -1.4%）；② 训练平台化（512c→512d 无增益，继续训练只是巩固错误目标）；
③ log 轴图全部失败（20/20 掩码差异 >8000px）。
**修复（2026-08-19）**：ChartDataset 改用与评估完全相同的代码路径构建轴映射
（detect_structure → read_ticks → build_axes → value_to_pixel 重建掩码），
代码级保证训练目标 ≡ 评估基准；旧 polyfit 保留为 fallback（检测失败时）。
实测 12ms/图（3500 张 ~42s 一次性缓存），pytest 106/106 通过。
**诊断工具**：scripts/eval_multi_diag.py（逐图/逐曲线 rel_rmse + 桶分类）、
scripts/diag_error_pattern.py（错误模式：local_spike 91 / systematic_bias 34 /
mixed 39 / partial_trace 3）、scripts/diag_pixel_bias.py（像素级偏差）、
scripts/diag_prob_peak.py（模型概率峰值偏移 +1~2px）。
**推理调优（2026-08-18）**：多曲线掩码阈值 0.5→0.3（model unsure 尾部如 prob 0.44
被恢复）+ 链尾部跳变截断（tracer 离开曲线时截断）→ 召回 67.7%（vs 66.7%）。
失败分类：175 条失败中 87 条 1-2% 边缘型（精度边际）、61 条 2-5%、27 条 >5%
（归属/追踪）。纯训练收益已平台化（512c 66.7% → 512d 64.1% 无增益）；
**突破需新方向**：更大模型（base 96）/更多曲线形态模板/图例颜色辅助归属。


**2026-08-20 会话（重训完成 + 评估 + 方案 C Phase 1）**：
- **重训完成**：evalfix.pt（epoch 4）续跑 21 epoch（共 25）→ **evalfix_r2.pt，val_iou 0.7636**
  （旧 512c 0.4993，+53%；epoch 4 暂停点 0.7345，+4%）。LR 重启扰动导致 epoch 2/7/15
  出现 val_iou 回落（最低 0.41），best-iou 保存策略保证 checkpoint 只升不降；最终
  EMA 在 lr→0 时收敛于 0.7636（epoch 17-21 连续创新高）。
- **评估发现回归**：evalfix_r2 + argmax 的 6a/6b = 0.6074/0.5889，低于 512c 基线
  （当前数据重跑 0.6741/0.6512）。逐图定位：回归集中在 marker_rich / multi_curve_
  comparison / three_stage 模板的**交叉密集区**（img_0069 右侧 x_px 774-905）——
  新模型交叉区双通道高概率，argmax 互斥更伤（列级验证：旧模型该处通道概率全零靠
  骨架连续性幸存，新模型通道偏移/丢失）；29 条曲线转好（log 轴/systematic 修复生效）。
- **顺带修复 bug**：_truncate_jumps 越界（dy[i+1] 在陡尾链上 IndexError，短路掩盖，
  新模型 7 图触发）→ +3 回归测试；pytest 113/113。
- **方案 C Phase 1（独立阈值多标签）实施**（commit 73fe133）：multi_independent_mask
  配置，交叉处两通道各自保留像素（LineFormer 方向）。val 集 6a/6b：
  - argmax：0.6074/0.5889（曲线数 163）
  - +independent：0.7222/0.6667（曲线数 137，虚检↑）
  - **+min_area 450：0.7185/0.7004（曲线数 168/180）—— 三项全面超越基线**（+4.4/+4.9pp）
- Phase D 验收（500+500 独立验收集）结果见 PHASE_D_ACCEPTANCE.md / 交接文档。

**2026-08-20 重大发现（增强 flip bug）**：_augment 的 cv2.flip(inst, 2) 对 (K,H,W)
掩码在 512 分辨率下被 OpenCV Python 绑定当作多通道图**只翻转高度轴**（垂直镜像、通道保留），
与图像的水平镜像错位——**历史全部训练 50% 样本目标被污染**（训练平台化 512c/512d 无增益、
val_iou 0.76 天花板可能与噪声目标有关）；768 分辨率下直接崩溃（末维 512 启发式上限）。
修复：numpy 水平镜像 + 通道反转（np.flip(inst,2)[::-1]），+2 回归测试（commit 9f7c66c）。
**云端 768 实验（2026-08-20 启动）将同时包含「修复」与「分辨率」两个变化，需 512-fixed
对照实验分离效应。**

**结论（2026-08-17 暂停）**：多曲线 K 通道方案方向正确（曲线数准确率 93%），
精度瓶颈 = 256 分辨率（3.3px 偏差 → rel 1-2%）；512 训练可解决（单曲线 512
模型 val_single 100%），但 512 训练 8GB 显存下 ~1h/epoch，需 6-10 小时。
**下一步建议**：512 微调完整训练（10 epoch，后台过夜）→ 预期召回 ≥90%。

## 六、研究依据（2026-08-16 复核 / 2026-08-19 扩充）
- LineFormer（arXiv:2305.01837）：折线图实例分割，逐实例像素回归（HuggingFace 权重
  t29mato/lineformer-battery-finetuned 可参考）；交叉像素多标签（Bresenham 3px
  重叠 GT）+ 匈牙利匹配；6a/6b 双口径评估（6a 只罚漏检、6b 罚虚检）——真实图
  UB-PMC 6a/6b = 93.1/88.25；社区微调经验：过检 +62.3% → 域内微调后 +9.6%
- ChartZero（arXiv:2605.05820，2026）：纯合成 10 万图零样本；**GOI 损失**（类内
  拉近 + 类间全局正交 + 小簇合并 IoU 0.75→0.82，专治交叉碎片）、CoordConv、
  VLM 掩码式图例匹配、ChartRM 端到端指标 0.921 vs GPT-4o 0.468
- Socratic Chart（arXiv:2504.09764）：掩码形态学细化（腐蚀/膨胀/高斯模糊）——推理后处理
- Efficient extraction of experimental data from line charts（Graphical Models 2025）：
  Mamba 增强 Transformer + 曲线掩码引导训练（+10%），YOLOv9 元素检测 + LSTM 刻度
  识别，PMC 转换精度 87.91%
- AI-ChartParser（CGF 2025）：多任务（元素+拐点+曲线）端到端 + 区间-均值数值映射
- Graph-FINDER（npj Comput. Mater. 2026）：免人工校准多线图数字化，nRMSE 0.018
- EpiCurveBench（medRxiv 2025）：100 张真实图，ECS/ERP 容错指标 + 虚检不罚/漏检
  记零的配对协议（最强模型仅 42.9%）——与 LineFormer 6b 口径一致，均提示「虚检
  惩罚过重会扭曲优化方向」
- 细长结构损失：clDice（arXiv:2003.07311）、Skeleton Recall Loss（ECCV 2024）、
  Boundary Loss——压断线/跳线/边界定位；白板笔画评价协议证明标准 IoU 掩盖薄线失败
- 追踪：Steger 亚像素脊线提取（±0.5px→±0.1px）、骨架度≥3 结点=交叉点 → 切分支 +
  Hungarian 配对（方向+颜色+置信度代价）、Dijkstra 全局最短路
- 训练策略：Focal Tversky（小结构 +25.7%）、EMA、copy-paste、STU-Net 宽度+深度
  联合缩放（base 64→128 是真实收益路径 2.1M→8.4M）、分辨率是细线任务首要杠杆
- WebPlotDigitizer：手动选线颜色分离——自动化版本即本方案颜色辅助（HSV 聚类图例区
  得参考色，交叉处按颜色距离重归属）
- 替代范式（远期）：SpatialEmbedding 判别式 embedding（任意曲线数、交叉天然分离，
  细线有风险）、Mask2Former 查询式、DETR 式查询（ChartZero 支持任意曲线数）
