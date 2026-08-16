# 下一步工作 Prompt — materials-curve-intel 项目（新对话交接版）

> AI 你好，这是项目阶段性交接文件。**你必须全文阅读后再开始工作。**
> 本文件可修改（不同于上级 Prompt.md）。
> 交接日期：2026-08-16（Phase A + Web 演示 + B-1 tick 读取加固 + **B-2 标题/单位识别完成 + B-3 坐标判别三信号 + B-4 YOLO 训练中**）

## 〇、先读这些（每次会话开始必读）

1. **`F:\CODE\New\Prompt.md`**（只读，禁止修改）— 项目唯一权威指导文件：
   竞赛目标（一等奖）、六大要求（前沿技术/效果第一/先请示/持怀疑态度调研/分工）、
   验收标准（RMSE ≤1% 满量程、坐标类型准确率 ≥98%、单图 <5s GPU、多曲线召回 ≥95%）、
   技术路线（YOLOv8-nano + PaddleOCR + U-Net + 坐标映射引擎）。
2. **`F:\CODE\New\baseline\README.md`** — baseline 完整架构/用法/评估/限制说明。
3. **`F:\CODE\New\baseline\AXIS_TEXT_RESEARCH.md`** — **坐标轴文字/数字高鲁棒识别
   调研与设计方案（下一阶段核心工作，必读）**，含真实 OCR 基线数据与失败样例诊断。
4. **`F:\CODE\New\baseline\TESTING_GUIDE.md`** — 用户实操测试指南。
5. **`F:\CODE\New\baseline\web\README.md`** — Web 演示系统说明。
6. 开始工作前：`cd F:\CODE\New\baseline && git log --oneline -25 && git status`。

## 一、项目背景与验收标准（摘要）

**主题：** 材料科学图像曲线智能识别与解析（自动从曲线图提取数据 → 结构化输出）。
**验收：** 曲线数据点 RMSE ≤ 坐标轴满量程 1%；坐标类型（线性/对数）判断准确率 ≥98%；
单张处理 <5s（GPU）；多曲线召回 ≥95%（Phase C）。
**验收方式：** ① dataset-platform 独立 seed 500 张测试图；② ≥50 张真实论文蠕变图
人工标注对比；③ 每个模块 ablation。
**首批领域：** 蠕变曲线（做透后再泛化）。

## 二、环境与仓库状态

**环境：** Anaconda 虚拟环境 **`mci`**（Python 3.11，RTX 4060 8GB / CUDA 12.6）
- torch 2.13.0+cu126（GPU ✓）、ultralytics 8.4.115（Phase B.5 用）
- paddlepaddle 3.3.1（**CPU 版**）+ paddleocr 3.7.0（真实 OCR ~6-15s/图）
- numpy **必须固定 1.26.4**；fastapi 0.141.1 / uvicorn 0.52.3（Web 已装）
- 其余见 `requirements.txt`

**仓库：** `F:\CODE\New\baseline`（git，master，18 个提交，HEAD=最新交接提交）
**归档：** `F:\CODE\New\baseline_backup_20260814_phaseA`（752MB，含 .git，
**截至 9d12bbc**；其后提交未归档——需要时复制为新目录 baseline_backup_<日期>）
**数据工厂：** `F:\CLAUDE\NewProject1\materials-curve-dataset-platform`（V0fix-final-2，
8 模板；**平台仓库零改动**，全部扩展在 baseline 内）

**数据目录（均不入库，gitignore）：**
| 目录 | 内容 |
|------|------|
| `data/synthetic` | 40 张合成测试集（seed 20260806，含 GT 侧车） |
| `data/train_synthetic` | 400 张旧训练集（gen_synthetic 工厂） |
| `data/train_platform` | **2000 张平台风格训练集**（8 模板×250、多曲线 1-5、log 轴 1096 张、含掩码/CSV/侧车/MCG-JSON/YOLO 标签） |
| `data/eval_platform` | **100 张单曲线平台评估集**（seed 20260816，可直接 evaluate） |
| `data/failures/` | **真实失败样本库**（`fail_001_axis_detection.png` = 用户提供，回归用） |
| `data/real_papers/` | 真实论文图收集区（README 指南已就位，待用户收集） |

**模型检查点（`models/checkpoints/`）：**
- `unet_curve.pt`（**默认** = 512 模型，val_iou 0.8166；已入库）
- `unet_curve_400baseline.pt`（旧模型备份，已入库）
- `unet_curve_256_v1/v2.pt`、`unet_curve_512_v1.pt`（中间产物，**本地未入库**，可删）

## 三、当前进展（已完成，2026-08-14）

### 3.1 提取管线（baseline，单曲线/单子图，接口稳定）
```
结构检测(CV) → 刻度OCR(PaddleOCR/Stub 双后端) → 坐标映射(线性/对数自动判别)
→ 曲线分割(U-Net 512 / CV 后备) → 骨架追踪+概率图亚像素细化
→ 像素→数据映射 → CSV/JSON/overlay 叠加图导出
```

### 3.2 数据与精度（Phase A 完成）
**`data/dataset_builder.py`（平台适配器，已入库）**：复用平台生成器 API，自渲染
保证 GT 像素级精确（buffer_rgba）；扩展：对数轴、退化、模板 settings 生效、
GBK 容错、YOLO bbox y 翻转。`--num-curves 1` 产可评估集。
**U-Net 重训链**：256（40+40ep）→ 512 微调（--init 30ep，AMP + 增强增强）。

**stub OCR 评估（512 模型，当前基准，不得回退）：**
| 评估集 | 旧模型（400 张合成） | **当前模型（2400 张混合 @512）** |
|--------|---------------------|----------------------------------|
| 合成 40 张 | med 0.77% / max 3.18% / 70% | **med 0.40% / max 1.65% / 87.5%** |
| 平台 100 张 | med 0.76% / max 5.65% / 61% | **med 0.20% / max 0.67% / 100%** |
| CV 后端（合成） | med 1.73% / max 82% / 40% | （未变，训练免后备） |

### 3.3 坐标轴文字/数字鲁棒性调研（下一阶段核心，详见 AXIS_TEXT_RESEARCH.md）

**真实 OCR（paddle）基线（平台集 100 张）——问题严重：**
- 硬失败 2/100（"0 readable ticks"）；98 张"成功"中 **中位 rel_rmse 0.43%、
  p90 39.7%、达标率仅 24.5%**；**坐标类型判错 56/98**（log 轴几乎全判成 linear）。

**失败样例诊断（用户图 `data/failures/fail_001_axis_detection.png`，已闭环）：**
- 整图 OCR 全部读出（标题/刻度/轴标题）→ **OCR 不是瓶颈**；
- 根因：`chart_structure.detect_structure` 的 y 轴线检测取"墨迹覆盖率 >45% 的
  **最左列**"→ 图左侧边缘墨迹被误判（y_axis_pixel=0）→ y 条带仅 5px 宽 →
  条带 OCR 空 + 标签分类条件 `center[0] < y_axis_col-2` 拒绝全部 y 标签；
  且该图 x/y 刻度标记全部未检出（"轴线旁 5px 带"假设刚性）。
- 推论：真实图左侧边缘残留是高频触发源；**短期 CV 加固 + 整图 OCR 兜底即可
  救活此类图；YOLO 检测是根治方案**。

**调研结论（论文+社区）**：图表文本识别学界范式 = 区域检测 + **文本角色分类**
（title/axis label/tick label/legend，见 ChartEye、ICDAR CHART-Infographics、
PlotQA）；坐标定标社区实践 = 轴端点/刻度锚点校准（WebPlotDigitizer/Engauge）；
刻度值必须做等差/等比序列校验；R² 双拟合单判据在刻度误读时失效。

### 3.4 Web 演示系统（展示版，已完成）
- `python web/app.py` → http://127.0.0.1:8000（自动开浏览器；`--port`/`--no-browser`）
- 批量多选/拖拽上传 → 串行提取 → 逐张结果卡片（overlay + 摘要 + 独立下载
  CSV/JSON/叠加图）；内置 6 张示例图已入库（`web/examples/`，不依赖 data/）；
  上传图无 GT 侧车时 stub 自动回退 paddle；启动后台预热 PaddleOCR。
- 限制：单曲线管线（多曲线只输出主曲线）；CPU OCR 慢。

### 3.6 Phase B-1 完成（tick 读取加固，2026-08-16）

**6 项改动全部落地（模块接口不变）：**
1. **整图 OCR 兜底**（tick_reader.py）：条带 OCR 有效刻度不足 → 整图 OCR + 宽松几何过滤 + 去重合并；
   兜底条件增强：读出 ≤2 但刻度标记 ≥4 也触发（救 img_0056 类漏读图）；
2. **轴线检测鲁棒化**（chart_structure.py）：y 轴线中央扫描（排除左右 3% 边缘）+ 与 x 轴线
   端点连通性校验；x 轴线排除底部边缘行（防边框误判）；轴线厚度动态估计（axis_top）；
3. **标签分类宽松化**（tick_reader.py）：区域规则（x：轴线下 + plot 宽 ±15%；y：plot 左 25%
   内 + 轴线以上 + 非标题区），不再依赖 y_axis_col 绝对边界；
4. **刻度带多尺度投票**（chart_structure.py）：2/5/8px 带 ≥2 尺度检出才计为刻度；
5. **映射校验**（coordinate_mapper.py）：RANSAC 双空间（linear/log10）剔除离群刻度（≥4 刻度）；
   端点锚点（0 起点）带**统计检验**（外推值须落在拟合 2σ 置信区间内才锚定，2 刻度用 0.5% span
   绝对阈值）——修复过程发现并避免了两类误锚（0.1 起轴、已有 0 刻度轴）；
6. **性能**：PaddleOCR 引擎类级缓存（(lang,device) 共享），消除每图重建模型 ~20s 浪费。

**验证结果：**
- pytest **77/77**（新增 17 项：RANSAC/锚点/分类/兜底/轴线/刻度投票回归测试）；
- 失败样例 **fail_001 恢复**（0 readable ticks → 1134 pts，x/y 全刻度读出，quality>0.9999）；
- stub 双路径**精确回退到基线**（合成 med 0.404%/87.5%；平台 med 0.204%/100% 达标）；
- paddle 平台 100 张：**硬失败 2→0；达标率 24.5%→48%（目标 90% 未达）；med 0.43%→0.14%；
  刻度识别率 x 97.2% / y 96.3%（核心目标达成）**；
- 新增指标：scripts/eval_axis_stats.py（刻度识别率/坐标类型准确率，读 report.csv）；
  evaluate.py 增加 n_ticks_x/y_gt、n_ticks_x/y_read 列（向后兼容）。

**剩余差距根因（B-3 范畴，已定位）：** 坐标类型判错（log 轴判 linear，x 64%/y 79% 准确率）——
2 刻度场景 R² 判据失效（任何模型完美拟合 → linear 胜出）+ 刻度值误读（0.1→100、上标丢失
"10¹"→"101"）+ RANSAC 选到误读一致子集。解法 = Phase B-3 三信号融合（刻度值序列校验等差/
等比 + 像素间距分析 + 轴标题先验），B-1 已为其铺好刻度读取基础。

### 3.5 测试与质量
- **60/60 pytest 通过**；已修复 bug 清单（勿回退）：DEFAULT_CONFIG_PATH 层级、
  _filter_mask_fragments（标题/边框碎片）、PaddleOCR 初始化锁、示例图 group 错位、
  gitignore 陷阱（`data/*`+`!data/*.py`；`!web/examples/*.png` 须指向文件）。
- git 历史 18 提交（Phase A.1 适配器 → A.2 重训+修复 → TESTING_GUIDE → Web →
  Web 批量+内置示例 → AXIS_TEXT_RESEARCH 调研）。

## 四、下一步计划（按优先级，每步先调研后动手、先请示用户再执行）

### Phase B-1：tick 读取加固（✅ 已完成 2026-08-16，见 §3.6；接口不变，pytest 89/89，
stub 双路径精确回退基线；paddle 达标率 24.5%→48%）

### Phase B-2：标题/轴标题/单位识别（✅ 已完成 2026-08-16，pytest 89/89）
- title_reader.py：区域+内容规则角色分类（title/x_label/y_label）；variable+unit
  提取（'Name (unit)' / 'Name / unit'）；log/ln 先验 → fit_axis kind_hint（B-3 信号 3）
- **竖排 y 轴标题旋转识别**（CW 旋转 90° 条带 + 2x OCR）——matplotlib +90° 旋转实测
  必须顺时针（CCW 读出碎片）
- extractor 接入：result.meta['titles'] 结构化输出；log_hint 传 build_axes
- dataset_builder：meta.json 新增 title/x_label/x_unit/y_label/y_unit GT 字段
- **验证（30 张平台集，paddle）：title/x_label/y_label 检出率 100%，variable/unit
  解析率 100%（目标 ≥90% 达成）**；验证脚本 scripts/eval_titles_batch.py +
  scripts/eval_title_stats.py

### Phase B-3：坐标类型判别三信号融合（✅ 已完成 2026-08-16，pytest 89/89）
- axis_kind.py 三信号投票：① 值序列一致性（等差/等比 + **'10N' 上标粘连重解析**：
  matplotlib log 标签 '10²'→OCR '102' → 重读 10^N；模糊序列（100/101/102/103 等差



### Phase B-4：YOLOv8-nano 结构检测（实施中，2026-08-16）
- train/train_detection.py 已创建（交接文件提及但原不存在）；数据 2000 张
  （6 类：plot_area/x_axis_line/y_axis_line/tick_label/legend_box/axis_title）
- 小批量 smoke 训练（500 张 40 epochs）后台运行中；后续接入 ChartStructure 输出
  + 文本区域精确定位


1. **整图 OCR 兜底**（✅ 已实施）：条带 OCR 为空 → 整图 OCR + 宽松几何过滤
   （失败样例整图 OCR 全对，此改动可救活该类图）；
2. **轴线检测鲁棒化**（chart_structure.py）：y 轴线不用"全局最左列"，改为
   中央扫描（排除左右 3% 边缘）+ 与 x 轴线端点连通性校验；x 轴线排除边缘行；
3. **标签分类宽松化**（tick_reader.py）：y 标签判定改用"plot 左 25% 宽度内、
   x 轴线以上、非标题区"区域规则，不依赖 y_axis_col 绝对边界；
4. **刻度带检测增强**：刻度带加宽 + 多尺度（2/5/8px）投票；
5. **映射校验 RANSAC**：fit_axis 前剔除离群刻度；轴端点锚点 fallback
   （刻度 <2 时用 1 刻度+端点定标，借鉴 WebPlotDigitizer）；
6. **验证**：失败样例恢复；合成 40 + 平台 100（stub + paddle 双路径）回归；
   新增"刻度文本识别率/坐标类型准确率"指标（paddle 路径当前达标率 24.5% 是
   主要攻坚对象，目标 ≥90%）。

### Phase B-2：标题/轴标题/单位识别（新能力，用户明确要求）
- 区域检测（几何带 + 后续 YOLO axis_title）→ OCR → 角色分类（位置+内容规则：
  "Creep strain (%)"→变量+单位；"log Time"→对数先验）→ 结构化输出到结果
  JSON/CSV 元数据；log 先验接入 fit_axis 的 kind_hint。
- 验证：合成/平台集 title/单位解析率 ≥90%（GT 对照）。

### Phase B-3：坐标类型判别三信号融合（目标 ≥98%）
- 刻度值序列校验（等差/等比，容差内）+ 像素间距分析（不依赖 OCR 值）+
  轴标题先验 → 投票定类型；R² 只作数值映射参数。
- 验证：合成 + 平台（含 2 刻度场景）轴类型准确率 ≥98%。

### Phase B-4：YOLOv8-nano 结构检测（根治结构检测，与 B-1~B-3 可并行）
- 训练数据已就绪：`data/train_platform/*_yolo.txt`（6 类，y 翻转已修正）；
- `train/train_detection.py`（ultralytics 已装）+ `models/detection/` +
  `configs/detection.yaml`；输出同一 `ChartStructure`；检测框给文本区域
  精确定位（tick_label/axis_title），文本识别仍走 PaddleOCR（检测识别解耦）；
- 小批量 500 张验证 mAP@0.5，再全量；对照 CV 检测 ablation。

### Phase A.3：真实论文图验证（与 B 并行，用户任务）
- 用户收集 ≥50 张真实蠕变图（先 10-15 张）→ `data/real_papers/raw/` +
  gold 标注（指南已就位）；AI 用 `--ocr paddle` 评估（真实图无 labels.json，
  不能走 stub）；失败图自动归入 `data/failures/` 做回归样本。

### Phase C（B 完成后再启动）：多曲线提取（验收：多曲线召回 ≥95%）
- U-Net K 通道/实例分割；组件评分 top-K + 颜色分离；legend_matcher.py
  （OCR 图例文本 + 颜色/线型关联）；多曲线评估指标（F1/召回、DTW）；
  数据已就绪（train_platform 多曲线 + `_curves.json` 清单）；
  Web 升级多曲线展示。

### Phase D：工程化与成果
- 测试报告（全部评估表）、500 张平台验收集（`dataset_builder --num-curves 1
  --count 500 --seed <新seed>`）、软著/专利/论文（按 Prompt.md 分工）。

## 五、已知的技术坑（务必先读，避免重复踩坑）

1. **Windows DLL 冲突**：torch 与 paddle 同进程互斥（WinError 127，双向）。
   OCR 用 CPU paddle（`enable_mkldnn=False` 必须设）；`tick_reader._ensure`
   先 import torch 再 import paddle（已加初始化锁）。
2. **numpy 版本**：mci 环境固定 **numpy==1.26.4**。
3. **matplotlib 渲染差异**：数据生成与评估必须同一环境；GT 像素用
   `fig.canvas.buffer_rgba()`，不要用 savefig 反推。
4. **Otsu 阈值失效**：浅色曲线（>150 灰度）；二值化用 `utils.ink_mask`。
5. **刻度带污染**：x 刻度带限 y 轴右侧；关联容差 25px（stub anchored 用标签中心）。
6. **网格去除**：覆盖率 >45% 且与刻度 ±1px 对齐才删（虚线免疫）；小核闭运算桥接。
7. **虚线曲线**：骨架端点桥接（≤48px）必须在组件选择前；训练掩码按实线绘制。
8. **U-Net 推理分辨率必须等于训练分辨率**（当前 512，configs/baseline.yaml：
   `segmenter: unet`、`unet_size: 512`；换模型用 `evaluate.py --unet-size N`）。
9. **评价口径**：rel_rmse 以 GT y 满量程归一；GT 稀疏点不能直接比像素。
10. **glob 过滤**：evaluate/run_baseline 必须过滤 `*_mask.png`。
11. **stub OCR 依赖 GT 侧车**（`*_labels.json`，仅示例图/数据集图有）；上传图
    选 stub 自动回退 paddle（Web 已处理）。
12. **标题文字碎片**：512 模型会把裁进 plot bbox 的标题笔画预测为曲线 →
    `_filter_mask_fragments` 骨架化前剔除（勿回退）。
13. **gitignore 陷阱**：父目录排除后子文件无法 re-include（`data/*`+`!data/*.py`；
    `!web/examples/*.png` 须指向文件）；中间检查点勿入库；`data/` 全量不入库。
14. **PowerShell 坑**：PS5.1 无 `Invoke-WebRequest -Form`（用 curl.exe/python）；
    后台跑服务器不要用 `Select-Object -First N` 截管道（会杀进程），重定向到文件。
15. **PaddleOCR 长条带**：超长条带（如 64×5913）会被 PaddleOCR 强制 resize 到
    max_side_limit 4000 内导致变形——条带裁剪必须合理（失败样例的根因之一）。

## 六、工作约定（继承 Prompt.md 六大要求）

- 每次会话开始重读 `Prompt.md` + 本文件 + `AXIS_TEXT_RESEARCH.md` + git log/status；
- 一切工作先调研（论文/官方文档/社区）再设计，**先请示用户再执行**；
- 每步改动跑 `python -m pytest tests -q` 与
  `scripts/evaluate.py --data-dir data/synthetic --ocr stub --segmenter unet` 回归，
  **指标不得回退**（stub 基准：合成 med ≤0.40%、平台 ≤1% 达标率 100%）；
  paddle 路径目标：达标率 24.5% → ≥90%（攻坚项）；
- 保持「模块接口稳定、逐步替换实现」；`data/` 不入库（代码例外），模型与
  训练数据本地；
- 定期归档：重大里程碑后复制 `baseline` 为 `baseline_backup_<日期>`（含 .git）。

## 七、常用命令速查（conda activate mci 后，工作目录 F:\CODE\New\baseline）

```bash
# Web 演示（自动开浏览器）
python web/app.py                      # 或 --port 8010 / --no-browser

# 平台适配器数据生成
python data/dataset_builder.py --out-dir data/train_platform --count 2000 \
    --seed 20260815 --yolo             # 训练集（多曲线）
python data/dataset_builder.py --out-dir data/eval_xxx --count 100 \
    --num-curves 1 --seed <新seed>     # 可评估单曲线集

# 旧合成工厂
python scripts/gen_synthetic.py --out-dir data/synthetic --count 40 --seed 20260806

# 评估（stub=确定性快；paddle=真实 OCR，慢）
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval \
    --ocr stub --segmenter unet
python scripts/evaluate.py --data-dir data/eval_platform --out-dir data/eval_plat \
    --ocr paddle --segmenter unet      # 真实 OCR 路径（当前达标率 24.5%，攻坚项）
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval_cv \
    --ocr stub --segmenter cv          # CV 后备对照

# 单图 + 调试（--debug 输出结构图/OCR 框/掩码图）
python scripts/run_baseline.py --image data/failures/fail_001_axis_detection.png \
    --out-dir data/outputs --ocr paddle --segmenter unet --debug

# U-Net 训练（--size 须与 unet_size 一致；--init 续训；--amp）
python train/train_segmentation.py --data-dir "data/train_platform,data/train_synthetic" \
    --val-dir data/synthetic --epochs 40 --batch 16 --amp \
    --out models/checkpoints/unet_curve.pt

# 测试
python -m pytest tests -q              # 当前 60/60
```

## 八、待用户确认/执行的事项

1. **确认 Phase B-3 启动**（坐标类型判别三信号融合：刻度值序列校验等差/等比 +
   像素间距分析 + 轴标题先验投票；paddle 达标率 48%→90% 的攻坚项，B-1 已定位
   失败根因并铺好刻度读取基础）；或先做 Phase B-2（标题/轴标题/单位识别）；
2. **真实论文图收集**（Phase A.3，用户任务）：≥50 张蠕变图，先 10-15 张；
   测试中再遇到失败图直接放入 `data/failures/` 或发路径；
3. 收集失败图（如 Web 上传测试失败）是回归样本的重要来源。
