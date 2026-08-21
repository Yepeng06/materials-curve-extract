# Phase D 验收材料清单（2026-08-19 起）

## 一、验收集（已生成，独立 seed 20260819）

| 集合 | 路径 | 数量 | 内容 |
|------|------|------|------|
| 多曲线 500 张 | data/eval_phased_500 | 500 | 曲线数 1-5（13/113/221/137/16），轴型 linear/linear 238、linear/log 104、log/linear 99、log/log 59，8 模板均衡 |
| 单曲线 500 张 | data/eval_phased_500_single | 500 | 全部单曲线，8 模板均衡，线性/对数混合 |

> 验收口径（Prompt.md）：曲线点 RMSE ≤1% 满量程；多曲线召回 ≥95%；
> 坐标类型 ≥98%；单图 <5s GPU。验收集生成器与训练/评估数据同源（dataset_builder），
> 但独立 seed，无重叠。

## 二、评估命令

```bash
PY=F:\anaconda3\envs\mci\python.exe

# 单曲线验收（Phase D）
$PY scripts/evaluate.py --data-dir data/eval_phased_500_single --out-dir data/eval_phased_single --ocr stub --segmenter unet

# 多曲线验收（Phase C 目标 ≥95% 召回，6a/6b 双口径）
$PY scripts/eval_multi_6ab.py --data-dir data/eval_phased_500 --model models/checkpoints/unet_multi_curve_evalfix.pt --out-dir data/eval_phased_multi --size 512
```

## 三、Ablation 矩阵（每模块单独评估）

| # | 模块 | 口径 | 对比 | 状态（2026-08-20） |
|----|------|------|------|------|
| A1 | 结构检测 | YOLO vs CV | mAP50；端到端 RMSE/召回 | 历史：yolo mAP50 0.942；未重跑 |
| A2 | 刻度 OCR | stub vs paddle | 刻度识别率、轴类型准确率 | 历史：paddle 99%；未重跑 |
| A3 | 轴型判型 | 三信号 vs 纯值序列 | 判型准确率（合成含 log 图） | 历史通过；未重跑 |
| A4 | 曲线分割 | U-Net vs CV | 单曲线 RMSE；多曲线召回 | 单曲线 Phase D 500/500、99.4% <=1%（unet_curve.pt） |
| A5 | 多曲线归属 | argmax vs independent 多标签 vs +嵌入合并 | 召回 6a/6b | **已做**：0.607→0.722→0.7185（+ma450）；嵌入合并待 E 模型 |
| A6 | 训练目标 | 旧 polyfit vs 评估一致路径（本修复） | val_iou、召回 | **已做**：512c 0.4993→evalfix_r2 0.7636（+53%） |
| A7 | 后处理 | 阈值 0.3 vs +跳变截断 vs +min-area | 召回/曲线数准确率 | **已做**：min_area450 6b 0.7004 历史最高 |
| A8 | 分辨率 | 256 vs 512 vs **768** | 召回、RMSE | **已做**：512 最佳（768 无端到端收益，2026-08-20 云端实测） |
| A9 | 损失 | BCE+Dice vs +骨架召回 vs **+GOI** | val_iou、薄线子集 | 骨架召回已随训；GOI 实现完成（3f08b3e），云端训练排队中 |

> 注：A1/A2/A3 有历史数据（B 系列），正式验收报告时整理入表。
## 四、真实图验证（待用户收集）

- data/real_papers/raw/：≥50 张蠕变论文图 + gold 标注 → paddle OCR 评估 + RMSE
- 失败图入 data/failures/ 回归

## 五、测试报告结构

1. 环境与复现（依赖版本、GPU）
2. 合成验收：单曲线（RMSE 分布、达标率）、多曲线（6a/6b 召回、曲线数准确率）
3. 真实图验收（待用户）
4. Ablation 表（上表 A1-A9）
5. 失败模式分析与剩余差距
6. 性能：单图耗时（GPU OCR 未启用前注明 CPU 瓶颈）

## 六、待办（验收前）

- [x] 验收集生成（500 多曲线 + 500 单曲线，独立 seed）
- [x] evalfix 训练完成（evalfix_r2，val_iou 0.7636）→ 全量评估进行中（A4/A6 主结果，2026-08-20）
- [x] A5 方案 C Phase 1 已实现（multi_independent_mask + min_area），val 集 6a/6b = 0.7185/0.7004 超基线
- [ ] 真实图收集（用户）
- [ ] A1/A2/A3 复跑记录（B 系列已有历史数据，整理入表）