# Phase C 根因修复与重训设计（2026-08-19）

## 一、根因（已证实，非猜测）

**训练目标 ≠ 评估基准**：train_segmentation_multi 的 `_fit_axis_from_labels`
用「标签中心 polyfit + 底部标签过滤 c[1] < img_h-85」重建轴映射，与评估侧
（detect_structure → read_ticks → build_axes）**轴型判定不一致**：

- 底部 y 刻度标签（如 0.001@527px，600px 图）被 515px 过滤排除后，
  剩余序列 0.1/1/10 比值=100 不触发 log 判定（阈值 >100）→ 判 linear；
- 评估侧保留底部标签 → 判 log。log 轴上 linear 拟合误差 100+px。
- **全数据核查：y 轴不一致 904/3548 (25.5%)、x 轴 65/3680 (1.8%)**，
  全部为 fit=linear vs GT=log。

**连锁效应（解释了全部主要失败模式）**：
| 现象 | 根因关联 |
|------|----------|
| systematic_bias 34 条（同图 4 曲线同向偏 -1.4%） | 模型学到错误轴型的目标 |
| 训练平台化（512c 66.7%→512d 64.1% 无增益） | 继续训练只是巩固错误目标 |
| log 轴图全失败（20/20 掩码差异 >8000px） | 掩码整体重建在错误映射下 |
| 模型概率峰值 +1~2px 系统偏移 | 部分源自目标偏移（待重训验证） |

## 二、修复（已实施并验证）

ChartDataset 改走**与评估完全相同的代码路径**构建轴映射
（detect_structure → read_ticks → build_axes → value_to_pixel 重建掩码），
代码级保证训练目标 ≡ 评估基准；旧 polyfit 保留为 fallback。

- 12ms/图（预构建缓存 ~42s/3500 张）
- **验证：240/240 val_multi 掩码通道与 GT 曲线像素重合 ≥95%（≤2px）**
- pytest 106/106；git e3a2d79

## 三、重训方案（需确认）

预期：修复 25% 训练图目标后，召回应从 67.7% 显著回升（尤其 log 轴图
与 systematic_bias 类）；继续沿用 512c 参数（base 64、BCE+Dice、Adam、
Cosine、AMP），从 512c 检查点续训 20-30 epoch（~3-5h，8GB batch 8）。

命令：
  $PY train/train_segmentation_multi.py --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single --val-dir data/val_multi,data/val_single --epochs 25 --batch 8 --size 512 --per-dir-limit 600 --init models/checkpoints/unet_multi_curve_512c.pt --out models/checkpoints/unet_multi_curve_evalfix.pt

训练后评估：$PY scripts/eval_multi.py --data-dir data/val_multi,data/val_single --model models/checkpoints/unet_multi_curve_evalfix.pt --out-dir data/eval_multi_evalfix --size 512

## 四、后续候选（重训后再定，按研究结论排序）

1. **GOI 损失（ChartZero arXiv:2605.05820）**：类内拉近 + 类间全局正交 +
   小簇合并——专治交叉碎片（消融 IoU 0.75→0.82）
2. **LineFormer 6a/6b 双口径 + 过检抑制**：虚检（19 张 pred=5）是独立
   失败源；IoU NMS/置信度截断
3. **clDice / Skeleton Recall Loss（ECCV 2024）**：细长结构损失，压断线/跳线
4. **Focal Tversky + EMA + copy-paste**：训练技巧（小结构 +25.7% 证据）
5. **曲线数语义先验（VLM）**：交叉归属与曲线数闭集化（PlotPick 召回
   88-96% 证据）——真实图阶段优先

## 五、诊断工具（本次新增，已入库）

- scripts/eval_multi_diag.py —— 逐图/逐曲线 rel_rmse + 桶分类 + 模板/形态归因
- scripts/diag_error_pattern.py —— 错误模式分类（local_spike/systematic_bias/...）
- scripts/diag_pixel_bias.py / diag_prob_peak.py —— 像素级/概率峰值偏移
- scripts/diag_mapping_consistency.py —— 训练/评估映射一致性检查
