# 下一步工作 Prompt — materials-curve-intel 项目（新对话交接版 v8 · 2026-08-26 归档）

> AI 你好，这是项目阶段性交接文件。**你必须全文阅读后再开始工作。**
> 本文件可修改（不同于上级 Prompt.md）。上一版 v7 见 git 历史。
> 交接日期：2026-08-26（**杠杆1+3 落地，val 6a 0.8481，Phase D 0.8641——历史最好**）
> 完整备份：`F:\CODE\New\baseline_backup_20260825\`（4.5GB，含 .git 与全部数据/模型）
> 用户计划在新对话中进行更深入的研究论证（论文/社区文档）以实现最终性能指标（6a 95%）。

---

## 〇、全局现状总结（先读这一节）

### 当前最优配置（已提交，默认）
- 模型：`models/checkpoints/unet_multi_chain.pt`（**杠杆1+杠杆3 训练产物**）
- 管线后处理：P1 条件 refine（接近区跳过细化）+ S1b 偏差校准（-0.67px）+ GOI 嵌入小簇合并 + min_area 450
- 配置：`configs/baseline.yaml`（multi_unet_checkpoint 已指向 chain）

### 指标总表（全部为 val 180 张 / Phase D 500 张独立验收集）

| 指标 | 项目起点（512c） | GOI | **当前（chain）** | 目标 |
|---|---|---|---|---|
| val 6a（纯召回） | 0.6741 | 0.7352 | **0.8481** | 0.95 |
| val 6b（罚虚检） | 0.6512 | 0.7192 | **0.8388** | — |
| >5% 桶 | ~37 | 31 | **18** | 0 |
| 1-2% 桶 | ~69 | 69 | **29** | — |
| 曲线数准确率 | 161/180 | 168/180 | **174/180** | — |
| **Phase D 6a** | — | 0.8065 | **0.8641** | 0.95 |
| Phase D 曲线数 | — | 346/500 | **399/500** | — |
| 单曲线 ≤1% 达标 | — | 99.4% | **99.6%** | 99.5%+ |
| 训练 val_iou | 0.4993 | 0.7740 | **0.7938** | — |
| pytest | 118/118 | 118/118 | **118/118** | 全绿 |

### 一句话技术结论
**从 0.6741 到 0.8481 的路径 = 训练目标逼近评估目标**：
1. **杠杆1（接近区对比损失）**：训练时对"曲线接近区"像素做 hinge 对比（拉向正确实例质心 + 推离错误实例）+ GOI 正交 0.1→0.3——>5% 桶首次下降（31→27）
2. **杠杆3（链级损失 v4）**：高斯带 L2 回归——模型逐列概率分布对齐 GT 曲线精确像素位置（根治 +0.67px 峰值偏移）——>5% 桶 25→18、1-2% 桶 34→29
3. **数据侧（P2/chain2）单独无效甚至有害**：困难样本训练 val_iou 升但端到端 6a 降（像素指标与端到端脱节，第三次印证）

---

## 一、时间线（2026-08-14 → 08-26，92 commits）

| 阶段 | 内容 | 关键结果 |
|---|---|---|
| Phase A（8/14） | 合成数据工厂 + U-Net 512 单曲线 + 管线 | 单曲线 val_iou 0.8166 |
| Phase B（8/15-16） | 刻度 OCR 加固（paddle 99%）、三信号判型、YOLOv8n 结构检测（mAP50 0.942） | B-5a 达标 |
| Phase C（8/17-18） | 多曲线 K=6 实例分割、CSV 重建掩码 | 512c val_iou 0.4993 |
| C-2 根因修复（8/19） | 训练/评估映射路径统一（25.5% log 轴训练图误标修复） | val_iou 0.7636（+53%） |
| 方案 E GOI（8/21-23） | 嵌入头 + 全局正交 + 小簇合并 | val_iou 0.7740，Phase D 6a 0.8065 |
| 第一批实验（8/23） | S3 轴校准加固、P1 条件 refine、S1b 偏差校准、R1e 真实图诊断 | 6a 0.7852；真实图 gap 量化（46% 成功率） |
| P2 困难样本（8/24） | 3000 张交叉样本重训 | val_iou 0.7754 但 6a 略降（否定） |
| **杠杆1（8/25）** | 接近区对比损失 + GOI 正交增强 | **6a 0.8185，>5% 桶首次降** |
| 杠杆1深化 zone2（8/25） | 对比 1.0/正交 0.5 | val 6a 0.8278 |
| **杠杆3 链级损失（8/25）** | 高斯带 L2 回归（v4 稳定版，v1-v3 崩溃迭代） | **val 6a 0.8481、Phase D 0.8641、>5% 桶 18** |
| chain2（8/25 晚） | 6000 张 hard2（含陡尾）+ chain_x | 曲线数 178（历史最高）但 6a 0.8389（存档变体） |

---

## 二、关键技术发现（正反结论，新对话勿重复踩坑）

### ✅ 有效
1. **训练目标必须与评估基准一致**（C-2 教训）：训练掩码重建与评估走同代码路径（detect_structure→read_ticks→build_axes→value_to_pixel）
2. **接近区对比损失**（杠杆1）：命中接近区嵌入无判别力根因（GOI 质心余弦 0.9996 被实证）
3. **链级损失 v4**（杠杆3）：高斯带 L2 回归，梯度随 p→1 衰减（稳定）；直接优化像素级位置精度
4. **偏差校准**（S1b）：+0.67px 系统性峰值偏移的常数校准（log 轴 1px≈2% rel 的缓解）
5. **接近区跳过窗口质心细化**（P1）：refine 在接近区被另一曲线概率拉偏

### ❌ 否定（记录在案）
1. **GOI 嵌入做归属裁决**：质心余弦 0.9996，无判别力
2. **重叠区 argmax 分离**：0.6907（交叉处剥像素）
3. **概率引导追踪**（全局/门控版）：0.7759/0.7778（接近区概率也无判别力）
4. **2× 上采样**：99.4%→98.6%（推理仍 resize 512 像素当量不变 + 判型翻转）
5. **base 96 冷启动**：9 epoch val 0.018（数据不足）
6. **768 分辨率**：无端到端收益（后处理尺度敏感）
7. **链损失 v1（质心 L1）**：背景概率拉偏质心崩；**v2（band-BCE）**：logit 爆炸崩；**v3（排重叠列）**：仍崩——只有 v4（带内 L2）稳定
8. **模型集成**（GOI+P2、chain+chain2 概率平均）：无增益（输出冗余）
9. **P2/chain2 数据扩充单独**：像素指标升但端到端降（hard 数据占比 36% 过度保守化）

### ⚠️ 工程坑（云端）
1. **云端容器内存**：cgroup 限制 ~128GB（memory.high 124GB）；chain/chain_x 原始分辨率 float32 缓存（13.8MB/张 × 2 × 3300 ≈ 91GB）会静默被杀（无 traceback）——**缓存必须存训练分辨率 512**
2. **base 64→96 续训**：全层 shape 不匹配（strict=False 对 shape 冲突仍 raise）——只能冷启动
3. **云端 SSH 会话**：setsid + nohup + </dev/null 保险；AutoDL 无卡/有卡切换会杀进程（checkpoint best-iou 兜底）
4. **PowerShell 转义**：autodl.py 命令避免内联引号，用脚本文件
5. **tar 上传**：从 baseline 目录打包会生成 data/data/ 嵌套（用 --strip-components=1 解压）
6. 数据准备期慢（3300 张 chain+chain_x 缓存 ~10 分钟）——非卡死，耐心等 epoch 1

---

## 三、当前代码/模型/配置状态

### 关键文件
| 文件 | 状态 |
|---|---|
| `train/train_segmentation_multi.py` | 含杠杆1（zone-* 参数）+ 杠杆3（chain-* 参数，y+x 双方向） |
| `src/mci/pipeline/curve_extractor.py` | P1 条件 refine + S1b bias 校准 |
| `src/mci/pipeline/chart_structure.py` | 朝内 y 刻度检测 |
| `src/mci/pipeline/coordinate_mapper.py` | S3 RANSAC tie-break 修复 |
| `src/mci/pipeline/bar_detector.py` | 柱状图检测（R1e 口径，初步） |
| `configs/baseline.yaml` | 默认 chain 模型 + 全部后处理参数 |
| `EXPERIMENTS_BATCH1_SUMMARY.md` | 全部实验记录 |
| `scripts/eval_real_zero_shot.py` | 真实图零样本诊断（断点续跑） |

### 模型检查点（models/checkpoints/）
- **unet_multi_chain.pt（默认，val_iou 0.7938）**
- unet_multi_chain2.pt（曲线数 178 变体）
- unet_multi_zone.pt / zone2.pt（杠杆1 系列）
- unet_multi_goi.pt（GOI 基线）、unet_multi_p2.pt（困难样本变体）
- unet_curve.pt（单曲线）

### 数据目录（data/，不入库）
- 训练：train_platform（2000）/ _4c / _5c / _single + train_platform_hard（3000 交叉）+ train_platform_hard2（6000 交叉+陡尾）
- 验证：val_multi（120）/ val_single（60）
- 验收：eval_phased_500（500 多曲线）/ _single（500 单曲线）
- 真实图：real_diag/（ChartQA 2565 + CHART-UB 5 + PMC 35；eval_subset.txt 100 张子集）
- 实验结果：data/experiments_*/（全部 JSON + 结论）

---

## 四、剩余差距与候选方向（新对话研究论证的重点）

**Phase D 6a 0.8641 → 95% 差 8.6pp；val 6a 0.8481 → 95% 差 10pp**

### 失败桶现状（val，chain 模型）
- >5% 桶 18 条（primary_obvious 12 为主——接近区归属残留）
- 2-5% 桶 34（low_quality_screenshot 11、multi_curve_comparison 8、primary_obvious 15）
- 1-2% 桶 29（log 轴像素精度残留）

### 候选方向（按证据强度）
1. **杠杆 1+3 联合调参网格**（contrast/margin/chain-weight/正交权重）——低成本 1-2 轮云端，预期 +1-2pp
2. **hard 数据比例控制**（chain2 教训：36% 过度保守；试 15-20% 比例）
3. **链级可微追踪**（训练直接优化追踪后链误差——工程量大，治本）
4. **x 方向 chain 权重单独调**（chain2 中 chain_x 与 y 同权重可能干扰）
5. **真实图验收**（用户 50 张蠕变图 + gold 标注；R1e 显示结构/OCR 是真实图第一瓶颈）
6. **柱状图检测完善**（R1e 口径；ChartQA plot_bbox 偏差问题待修）

### 坦白说明
合成集 6a 95% 是否可达到存疑：文献中（LineFormer/ChartZero/EpiCurveBench）无专用方法达到 95% 级多曲线召回。**最终验收口径需要与用户对齐**（合成独立集 vs 真实图、6a vs 6b、含曲线数惩罚与否）。

---

## 五、常用命令速查

```bash
PY=F:\anaconda3\envs\mci\python.exe
cd F:\CODE\New\baseline

# 评估（当前默认 chain 模型）
$PY scripts/eval_multi_6ab.py --data-dir data/val_multi,data/val_single --out-dir data/eval_x --size 512
$PY scripts/eval_multi_6ab.py --data-dir data/eval_phased_500 --out-dir data/eval_x --size 512
$PY scripts/eval_multi_diag.py --data-dir data/val_multi,data/val_single --out-dir data/eval_x --size 512

# 训练（云端 4090，batch 16；本地 batch 8）
$PY train/train_segmentation_multi.py --data-dir <6目录> --val-dir data/val_multi,data/val_single \
  --epochs 25 --batch 16 --size 512 --per-dir-limit 600 \
  --init models/checkpoints/unet_multi_chain.pt --out models/checkpoints/<name>.pt \
  --zone-contrast 1.0 --zone-dist 4 --zone-margin 0.3 --goi-ortho 0.5 --chain-weight 0.15

# 云端（F:\CODE\New\autodl.py + autodl_secret.json + autodl_download.py）
$PY autodl.py run "..." / upload / upload_dir / download

# 测试
$PY -m pytest tests -q   # 118/118
```

---

## 六、云端状态与提醒
- **AutoDL 4090 实例训练已完成——请关机（¥1.98/h）**
- 云端 /root/baseline 有最新代码与数据（hard/hard2 已传）
- 无卡模式 SSH 可用（传数据）；有卡模式才能训练
- autodl_secret.json 不入库（密码变更只改此文件）

---

> 交接 v8 完成于 2026-08-26。详细实验记录见 EXPERIMENTS_BATCH1_SUMMARY.md（含全部正反结论）；备份 baseline_backup_20260825。
