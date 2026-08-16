# 下一步工作 Prompt — materials-curve-intel 项目（新对话交接版）

> AI 你好，这是项目阶段性交接文件。**你必须全文阅读后再开始工作。**
> 本文件可修改（不同于上级 Prompt.md）。
> 交接日期：2026-08-14（Phase A 完成 + Web 演示版上线后）

## 〇、先读这些（每次会话开始必读）

1. **`F:\CODE\New\Prompt.md`**（只读，禁止修改）— 项目唯一权威指导文件：
   竞赛目标（一等奖）、六大要求（前沿技术/效果第一/先请示/持怀疑态度调研/分工）、
   验收标准（RMSE ≤1% 满量程、坐标类型准确率 ≥98%、单图 <5s GPU、多曲线召回 ≥95%）、
   技术路线（YOLOv8-nano + PaddleOCR + U-Net + 坐标映射引擎）。
2. **`F:\CODE\New\baseline\README.md`** — baseline 完整架构/用法/评估/限制说明。
3. **`F:\CODE\New\baseline\TESTING_GUIDE.md`** — 用户实操测试指南（含验收口径）。
4. **`F:\CODE\New\baseline\web\README.md`** — Web 演示系统说明。
5. 开始工作前：`cd F:\CODE\New\baseline && git log --oneline -20 && git status`。

## 一、项目背景与验收标准（摘要）

**主题：** 材料科学图像曲线智能识别与解析（自动从曲线图提取数据 → 结构化输出）。
**验收（当前最接近的量化项）：** 曲线数据点 RMSE ≤ 坐标轴满量程 1%；坐标类型
（线性/对数）判断准确率 ≥98%；单张处理 <5s（GPU）；多曲线召回 ≥95%（Phase C）。
**验收方式：** ① dataset-platform 独立 seed 500 张测试图；② ≥50 张真实论文蠕变图
人工标注对比；③ 每个模块 ablation。
**首批领域：** 蠕变曲线（做透再做应力-应变、极化曲线）。

## 二、环境与仓库状态

**环境：** Anaconda 虚拟环境 **`mci`**（Python 3.11，RTX 4060 8GB / CUDA 12.6）
- torch 2.13.0+cu126（GPU ✓）、ultralytics 8.4.115（Phase B 用）
- paddlepaddle 3.3.1（**CPU 版**）+ paddleocr 3.7.0（真实 OCR ~6-15s/图）
- numpy **必须固定 1.26.4**；fastapi 0.141.1 / uvicorn 0.52.3（Web 已装）
- 其余见 `requirements.txt`

**仓库：** `F:\CODE\New\baseline`（git，master，17 个提交，HEAD=c236c17，工作区干净）
**归档：** `F:\CODE\New\baseline_backup_20260814_phaseA`（752MB，含 .git，**截至 9d12bbc**；
web 相关 5 个提交未归档——如需再归档请复制为新目录，例如 baseline_backup_20260815_web）
**数据工厂：** `F:\CLAUDE\NewProject1\materials-curve-dataset-platform`（V0fix-final-2，
8 模板 YAML；**平台仓库零改动**，全部扩展在 baseline 内）

**数据目录（均不入库，gitignore）：**
| 目录 | 内容 |
|------|------|
| `data/synthetic` | 40 张合成测试集（seed 20260806，含 GT 侧车） |
| `data/train_synthetic` | 400 张旧训练集（gen_synthetic 工厂） |
| `data/train_platform` | **2000 张平台风格训练集**（8 模板×250、多曲线 1-5、log 轴 1096 张、含掩码/CSV/侧车/MCG-JSON/YOLO 标签） |
| `data/eval_platform` | **100 张单曲线平台评估集**（seed 20260816，可直接 evaluate） |
| `data/real_papers/` | 真实论文图收集区（README 指南已就位，待用户收集） |

**模型检查点（`models/checkpoints/`）：**
- `unet_curve.pt`（**默认** = 512 模型，val_iou 0.8166；已入库）
- `unet_curve_400baseline.pt`（旧模型备份，已入库）
- `unet_curve_256_v1.pt / _v2.pt / _512_v1.pt`（中间产物，**本地未入库**，可删）

## 三、当前进展（已完成，2026-08-14）

### 3.1 提取管线（baseline，单曲线/单子图，接口稳定）
```
结构检测(CV) → 刻度OCR(PaddleOCR/Stub 双后端) → 坐标映射(线性/对数自动判别)
→ 曲线分割(U-Net 512 / CV 后备，--segmenter 切换) → 骨架追踪+概率图亚像素细化
→ 像素→数据映射 → CSV/JSON/overlay 叠加图导出
```

### 3.2 数据与精度（Phase A 完成）
**`data/dataset_builder.py`（平台适配器，已入库）**：复用平台生成器 API
（sample_parameters / generate_curve_data / MCG-JSON / YOLO 收集），自渲染保证
GT 像素级精确（buffer_rgba）；扩展：对数轴、退化、模板 settings 生效、GBK 模板
容错、inside_lower_left 图例修正、YOLO bbox y 翻转。`--num-curves 1` 产可评估集。

**U-Net 重训链**：256（40ep → 续 40ep）→ 512 微调（--init 续 30ep，AMP +
增强增强：随机裁剪/透视/扫描噪声）。**最终评估（stub OCR，512 模型）：**

| 评估集 | 旧模型（400 张合成） | **当前模型（2400 张混合 @512）** |
|--------|---------------------|----------------------------------|
| 合成 40 张 | med 0.77% / max 3.18% / 70% | **med 0.40% / max 1.65% / 87.5%** |
| 平台 100 张 | med 0.76% / max 5.65% / 61% | **med 0.20% / max 0.67% / 100%** |
| CV 后端（合成） | med 1.73% / max 82% / 40% | （未变，训练免后备） |

→ 已达成"中位 ≤0.5%"目标；平台集全部 ≤1%。合成集剩余 >1% 样本为极端动态范围
曲线（线性 y 跨 4.6 个数量级，如 img_0022/img_0004），可作后续攻坚点。

### 3.3 Web 演示系统（展示版，已完成）
- **启动：** `conda activate mci && cd F:\CODE\New\baseline && python web/app.py`
  （http://127.0.0.1:8000，自动开浏览器；`--port`/`--no-browser` 可选）
- **功能：** 上传 PNG/JPG（**批量多选/拖拽，串行提取、逐张状态徽标与结果卡片**）→
  提取 → overlay 叠加图 + 摘要（曲线数/轴类型/耗时/警告）→ 每张卡片独立下载
  CSV / JSON / 叠加图；示例图一键测试（**内置 6 张入库于 `web/examples/`，不依赖
  data/**；data/ 存在时额外列出）；上传图无 GT 侧车时 stub 自动回退真实 OCR 并提示；
  启动后台预热 PaddleOCR。
- **限制：** 单曲线管线（多曲线只输出主曲线，Phase C 后升级）；CPU OCR 慢。

### 3.4 测试与质量
- **60/60 pytest 通过**（含适配器格式契约测试、碎片过滤单测、端到端）
- 本阶段修复的 bug（勿回退）：`DEFAULT_CONFIG_PATH` 目录层级（默认配置此前从未
  生效）；`_filter_mask_fragments`（标题/边框碎片污染追踪，img_0033 19.1%→0.33%）；
  PaddleOCR 初始化锁（Web 多线程）；示例图 group 字段错位（前端点击 404）；
  gitignore 父目录排除陷阱（`data/*` + `!data/*.py`；`!web/examples/*.png` 须指向文件）。
- git 历史 17 提交：Phase A.1 适配器 → A.2 重训+修复 → TESTING_GUIDE → Web 系统 →
  Web 批量+内置示例。

## 四、下一步计划（按优先级，每步先调研后动手、先请示用户再执行）

### Phase A.3：真实论文图验证（建议先做，直接支撑"验收方式 2"）
1. **用户收集 ≥50 张真实蠕变论文图**（先 10-15 张即可开始），放
   `data/real_papers/raw/`，gold set 标注指南见 `data/real_papers/README.md`；
2. AI 侧准备：写真实图评估脚本（`--ocr paddle` 路径；CPU 慢，50 张约 10-20 分钟，
   可后台跑）；注意真实图通常没有 `*_labels.json` 侧车 → **不能用 stub**；
3. 若真实图无 GT，先人工抽查 overlay 效果；gold 标注 10-15 张后跑 rel_rmse 统计；
4. 真实图暴露的失败模式（扫描噪声、多子图、低分辨率）列入 Phase B/C 输入。

### Phase B：坐标与结构升级（对应技术路线）
5. **坐标类型判别加强**（当前靠 ≥3 刻度拟合残差 R²）：调研后补充
   「刻度间距分析 + 轴标题文本先验（OCR 读 "log Time" 等）」，目标准确率 ≥98%；
   评估口径：合成 40 + 平台 100 张的轴类型准确率（当前 ~100%，重点测真实图）。
6. **YOLOv8-nano 结构检测**（替换 `chart_structure.py` 的 CV 检测）：
   - 训练数据已就绪：`data/train_platform/*_yolo.txt`（6 类：plot_area/
     x_axis_line/y_axis_line/tick_label/legend_box/axis_title，y 翻转已修正）；
   - 写 `train/train_detection.py`（ultralytics 已装）+ `models/detection/` +
     `configs/detection.yaml`；输出同一 `ChartStructure` 数据类，管线其余不动；
   - 先小批量（如 500 张）验证 mAP@0.5，再全量；对照 CV 检测做 ablation。

### Phase C：多曲线提取（验收项：多曲线召回 ≥95%）
7. **多曲线分割**：U-Net 输出 K 通道或实例分割；组件评分逻辑改 top-K + 颜色分离；
   数据已就绪（`data/train_platform` 2000 张多曲线图 + 每曲线 CSV + `_curves.json`
   清单；掩码目前是"全曲线合一"，实例掩码需在适配器加 `generate_instance_mask`
   模式或后处理拆分）。
8. **图例匹配**：实现 `legend_matcher.py`（OCR 图例文本 + 曲线颜色/线型关联），
   填充 `Curve.legend_label`（协议已在 `pipeline/legend_matcher.py` 定义）。
9. **多曲线评估指标**：扩展 `eval/metrics.py`（多曲线 F1/召回、DTW）；建
   `data/eval_platform_multi`（适配器 `--num-curves 2-4` 生成）作为评估集。
10. **Web 升级**：多曲线结果展示（每曲线独立下载）、批量并发选项。

### Phase D：工程化与成果（竞赛交付）
11. 测试报告（汇总全部评估表）、软著材料、专利交底书框架、论文大纲
    （按 Prompt.md 第九、十节分工：AI 写技术部分，用户填个人信息/背景）。
12. 500 张平台测试集（验收方式 1）：`data/dataset_builder.py --num-curves 1
    --count 500 --seed <新seed>` 生成后 evaluate，作为正式验收报告数据。

## 五、已知的技术坑（务必先读，避免重复踩坑）

1. **Windows DLL 冲突**：torch 与 paddle 同进程互斥（WinError 127，双向）。
   方案：OCR 用 CPU paddle（`enable_mkldnn=False` 必须设）；`tick_reader._ensure`
   里**先 import torch 再 import paddle**（已加初始化锁，Web 多线程安全）。
2. **numpy 版本**：mci 环境固定 **numpy==1.26.4**（paddle 3.3 与 numpy 2.x 不兼容）。
3. **matplotlib 渲染差异**：数据生成与评估必须同一环境；GT 像素用
   `fig.canvas.buffer_rgba()`，**不要用 savefig 反推**。
4. **Otsu 阈值失效**：浅色曲线（>150 灰度）判为背景；二值化用 `utils.ink_mask`。
5. **刻度带污染**：x 刻度带限 y 轴右侧；刻度-标签关联容差 25px（stub anchored=True
   直接用标签中心像素）。
6. **网格去除**：行/列覆盖率 >45% 且与刻度 ±1px 对齐才删（虚线免疫）；Hough 按截距
   聚类防误删；删除后小核闭运算桥接。
7. **虚线曲线**：骨架端点「方向对齐+异组件+端点仅用一次」桥接（≤48px），必须在组件
   选择前；训练掩码按实线绘制。
8. **U-Net 推理分辨率必须等于训练分辨率**：当前 512。默认配置已生效
   （configs/baseline.yaml：`segmenter: unet`、`unet_size: 512`）。评估 512 外的模型
   用 `scripts/evaluate.py --unet-size N` 或 config 覆盖。
9. **评价口径**：`rel_rmse` 以 GT y 满量程归一；GT 稀疏采样点不能直接比像素
   （陡峭段错位），用稠密链最近邻/插值。
10. **glob 过滤**：evaluate.py / run_baseline.py 必须过滤 `*_mask.png`（勿回退）。
11. **stub OCR 依赖 GT 侧车**（`*_labels.json`，仅示例图/数据集图有）；任意上传图
    选 stub 会自动回退 paddle（Web 已处理）。
12. **标题文字碎片**：512 模型会把裁进 plot bbox 的标题笔画预测为曲线碎片 →
    `curve_extractor._filter_mask_fragments` 在骨架化前剔除（勿回退）。
13. **gitignore 陷阱**：父目录被排除后子文件无法 re-include（`data/*` + `!data/*.py`；
    `web/examples` 的例外必须写成 `!web/examples/*.png`）；中间检查点勿入库
    （本地保留即可）；`data/` 全量不入库。
14. **PowerShell 坑**：PS5.1 无 `Invoke-WebRequest -Form`（用 curl.exe 或 python）；
    后台跑服务器**不要**用 `Select-Object -First N` 截管道（会杀进程），重定向到文件。

## 六、工作约定（继承 Prompt.md 六大要求）

- 每次会话开始重读 `Prompt.md` + 本文件 + `git log`/`git status`；
- 一切工作先调研（论文/官方文档/社区）再设计，**先请示用户再执行**；
- 每步改动跑 `python -m pytest tests -q` 与
  `scripts/evaluate.py --data-dir data/synthetic --ocr stub --segmenter unet` 回归，
  **指标不得回退**（当前基准：合成 med ≤0.40%、平台集 ≤100% 达标）；
- 保持「模块接口稳定、逐步替换实现」；`data/` 不入库（代码例外），模型检查点
  与训练数据本地；
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

# 评估（stub=确定性；paddle=真实 OCR）
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval \
    --ocr stub --segmenter unet
python scripts/evaluate.py --data-dir data/eval_platform --out-dir data/eval_plat \
    --ocr stub --segmenter unet
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval_cv \
    --ocr stub --segmenter cv         # CV 后备对照

# 单图 + 调试
python scripts/run_baseline.py --image data/eval_platform/img_0000.png \
    --out-dir data/outputs --ocr stub --segmenter unet --debug

# U-Net 训练（--size 512 须与 unet_size 一致；--init 可续训）
python train/train_segmentation.py --data-dir "data/train_platform,data/train_synthetic" \
    --val-dir data/synthetic --epochs 40 --batch 16 --amp \
    --out models/checkpoints/unet_curve.pt

# 测试
python -m pytest tests -q              # 当前 60/60
```

## 八、待用户确认/执行的事项

1. **真实论文图收集**（Phase A.3，用户任务）：≥50 张蠕变图，先 10-15 张；
   指南 `data/real_papers/README.md`；
2. 新对话开始后请用户确认 Phase 顺序：建议 A.3（真实图，可并行收集）→ B.5
   （YOLO 检测，数据已就绪）或 C（多曲线）二选一推进；
3. Web 演示已可用于答辩演示；如需绑定"展示版"再归档一次请告知。
