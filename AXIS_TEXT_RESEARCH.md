# 坐标轴文字/数字高鲁棒识别 — 调研与设计方案（Phase B）

> 状态：调研完成，方案待用户确认后实施。日期：2026-08-14。
> 问题背景：当前管线在真实曲线图上出现
> `y-axis: only 0 readable ticks (need >= 2)` 类失败；且标题/轴标题/单位
> 目前完全未被识别。本阶段目标：**高鲁棒地识别刻度数字、标题、坐标轴文字、
> 单位**，覆盖各类图片。

## 一、现状（代码事实）

当前 tick 读取链路（`pipeline/tick_reader.py` + `chart_structure.py`）：

```
CV 刻度标记检测（轴线旁 5px 条带找短笔画）
  → OCR 只裁两个条带（x：轴线下方 1/8 图高；y：轴线左侧竖带），放大 2x
  → PaddleOCR(PP-OCRv4 mobile, lang=en) 识别文本
  → parse_number_text 解析成数值（支持科学计数法/上标/千分位/%）
  → 标签按位置分类（x：中心在 x 轴线下方；y：中心在 y 轴线左侧）
  → 与刻度标记全局最近配对（tol=25px），无标记则用标签中心生成 tick
  → fit_axis：线性/对数双拟合，R² 判据自动判别坐标类型
```

**"0 readable ticks" 的直接含义**：该轴上 `value is not None` 的 tick 为 0，
即 OCR 在条带内一个可解析文本都没读到（或全被位置分类排除）。上游 `build_axes`
要求每轴 ≥2 个有效刻度。

**标题/轴标题/单位现状**：`ChartStructure` 只记录 axis_title 的 bbox（YOLO
标签有该类），OCR 不读、坐标判断不使用——完全是空白能力。

## 二、文献与社区调研摘要

### 2.1 学术方法（图表文本提取）
- **ChartEye**（arXiv:2408.16123, IEEE 10410945）：深度图表信息提取框架，
  将图表理解组织为多任务——元素检测 + 文本提取 + **文本角色分类**
  （title / axis label / tick label / legend / data label）。这是当前学界
  处理"图表文字"的标准范式：**先检测文本区域，再按角色分类**，而不是纯几何裁剪。
- **ICDAR 2019 CHART-Infographics 竞赛**（IEEE 8978105）：明确提出从
  infographic 中提取结构化图表数据，文本按角色（title、x/y axis label、
  tick label、legend、data value）标注与评估——角色分类是基准任务。
- **DeMatch**（ICDAR 2021, DOI 10.1007/978-3-030-86334-0_45）：图表面板
  理解，强调文本上下文（上下文感知的元素关联）。
- **PlotQA 数据集**（github.com/NiteshMethani/PlotQA）：为图表 QA 提供
  **text role labels**（title/axis tick labels/axis labels/data labels），
  是文本角色分类训练/评测的事实标准之一。
- **Generalization of Fine Granular Extractions from Charts**（ICDAR 2023,
  DOI 10.1007/978-3-031-41679-8_6）：图表细粒度提取（含文本）的泛化性研究，
  指出合成数据训练 → 真实图泛化需要多样性覆盖（与本项目合成数据路线一致）。

### 2.2 社区工具（坐标校准实践）
- **WebPlotDigitizer**（automeris.io / DeepWiki 4.3 Coordinate Systems &
  Axes）：坐标映射采用**用户点选两个/三个校准点**（轴端点或已知刻度）+ 支持
  线性/对数/极坐标；自动刻度检测仅作辅助。其成熟实践说明：**刻度锚点的数量
  与质量决定映射精度**；少刻度时用轴端点+单个刻度也能定标（本项目 ≥2 刻度
  的要求偏严，可放宽为"轴端点锚点"机制）。
- **Engauge Digitizer**（github.com/akhuettel/engauge-digitizer）：同类半
  自动工具，支持"自动检测+人工修正"，强调交互式校验。
- **结论**：全自动场景下，社区工具也承认刻度识别的失败率，普遍靠人工点选
  兜底。本项目是"全自动竞赛系统"，故需**多重信号冗余**（几何 + OCR +
  结构先验），而非依赖单一信号。

### 2.3 OCR 工程实践
- PaddleOCR（PP-OCRv4 mobile det/rec）文本检测对 8-16px 小字（刻度标签典型
  尺寸）识别率下降；常见补救：**条带放大 2-4x、CLAHE 对比度归一化、多尺度
  投票**（社区共识，DeepWiki/CSDN 大量工程实践）。
- 刻度文本的领域解析（科学计数法、上标、单位符号）需要**领域词典 + 规则
  解析**（本项目 `parse_number_text` 已实现主体，需扩展脏文本容错）。

## 三、失败模式分析（"0 readable ticks" 候选根因）

### 3.0 真实 OCR 基线（2026-08-14，平台集 100 张，paddle + U-Net）

> 用当前管线跑真实 OCR 路径得到的关键基线数据，证明问题的严重程度：

- **硬失败**：2/100 张报 "only 0 readable ticks" 类错误；
- **软失败（更普遍）**：98 张"成功"中 **中位 rel_rmse 0.43%、p90 39.7%、
  最大 108%、达标率仅 24.5%**；
- **坐标类型判错 56/98**：几乎所有 `log` 轴被误判为 `linear`
  （`gt=(*,log) → pred=(*,linear)`、`gt=(log,log) → pred=(linear,linear)`）。

**机制分析（对应判据漏洞）：**
1. log 轴刻度值是小数字（0.01/0.1/1/10），条带小字 + 2x 放大下 OCR 漏读/误读率高；
2. 有效刻度 <2 → 硬失败；值被误读（如 0.01→0.1 或 0.1→1）→ log 拟合 R² 崩塌；
3. 只剩 2 个有效刻度时**任何模型都完美拟合（R²=1）**，`fit_axis` 判据让 linear 胜出；
4. 刻度像素等距时线性拟合 R² 也较高，`r2_log > r2_lin + 0.01` 不成立 → linear 胜出。
→ **R² 双拟合单判据对"刻度值被读错"毫无抵抗力**；且无"刻度值必须是等差/等比
   序列"的规则校验，误读不报错直接进入映射。

### 3.2 失败样例诊断（2026-08-14，用户提供图 `data/failures/fail_001_axis_detection.png`）

**现象**：`y-axis: only 0 readable ticks (need >= 2)`。
**诊断结论（已闭环）**：
- 整图 OCR（PaddleOCR）**全部读出**：标题"材料拉伸实验图"、y 刻度 60/50/40/30/20/10/0、
  x 刻度 0/10/20/30/40/50、轴标题"X轴/Y轴" → **OCR 不是瓶颈**；
- **根因在结构检测 `chart_structure.detect_structure`**：
  1. y 轴线检测取"墨迹覆盖率 >45% 的**最左列**"（`cols[0]`）→ 该图最左列
     （x=0）有边缘墨迹被误判为 y 轴线（`y_axis_pixel=0`）；
  2. y 条带宽度 = `y_axis_col + 5 = 5px` → 条带 OCR 区域为空 → 0 个文本；
  3. 即便有文本，标签分类条件 `center[0] < y_axis_col - 2` 在 `y_axis_col=0`
     时拒绝全部 y 标签（`center[0] < -2` 恒假）；
  4. 该图 x/y 刻度标记也全部未检出（`x_ticks_px=[]`、`y_ticks_px=[]`，
     刻度样式/位置超出"轴线旁 5px 带"的刚性假设）。
- **推论**：真实图左侧边缘残留（截图边框/裁剪痕）是"0 readable ticks"的高频
  触发源；结构检测的"全列覆盖率 + 最左列"与"5px 刻度带"假设对真实布局过于
  刚性。**短期 CV 加固 + 整图 OCR 兜底即可救活此类图；YOLO 检测是根治方案。**

### 3.3 层 1 修复优先级（基于 3.2）

1. **整图 OCR 兜底**（最高优先，立即生效）：条带 OCR 为空时降级整图 OCR +
   宽松几何过滤（该样例整图 OCR 全对）；
2. **轴线检测鲁棒化**：y 轴线不用"全局最左列"，改为中央扫描
   （排除左右 3% 边缘）+ 与 x 轴线端点连通性校验；x 轴线同样排除边缘行；
3. **标签分类宽松化**：y 标签判定不依赖 `y_axis_col` 绝对边界，改用
   "plot 左 25% 宽度内、x 轴线以上、且非标题区"的区域规则；
4. **刻度带检测增强**：刻度带加宽 + 多尺度（2px/5px/8px 带）投票，或对
   无刻度场景直接用"标签中心生成 tick"（已有 fallback，需确保标签先被读到）。

### 3.1 "0 readable ticks" 候选根因（诊断前假设清单）

| # | 环节 | 根因假设 | 证据/验证方法 |
|---|------|----------|----------------|
| 1 | OCR 条带裁剪 | y 轴标签条带宽度 `y_axis_col+5` 依赖 CV 轴线检测；轴线检测偏右/偏左 → 条带不覆盖标签 | debug 输出条带图（`--debug` 已存 `*_ocr.png`） |
| 2 | OCR 识别 | 真实图小字（<10px）、JPEG 噪声、旋转、衬线字体 → PP-OCRv4 漏检/误检 | 条带放大 4x 后人工核对识别文本 |
| 3 | 文本解析 | 识别脏文本（`1.5x103`、`10^ 3`、`O` 混淆 `0`）→ parse 失败 | 记录"OCR 读到但 parse 失败"的文本样本 |
| 4 | 位置分类 | 标签中心刚好越界（y1+10、x0-40 边界）→ 被过滤 | debug 绘制分类前后框 |
| 5 | 刻度标记 | 标记检测失败本身不致命（有 fallback），但若条带空 → 双双失败 | — |
| 6 | **旋转文本** | 论文图 y 轴刻度标签常为**竖排/旋转 90°**；当前 `use_textline_orientation=False` 且 y 条带窄 → 旋转标签漏检 | 失败样例图目视确认；开启方向分类或对 y 条带旋转 90° 后识别 |

> 实测结论（3.2）：根因 #1+#4 被确认（轴线检测错 → 条带空 + 分类全拒）；
> #2 未命中（该样例整图 OCR 全对）；#6 在真实论文图中仍高频，需继续覆盖。

## 四、设计方案（分层加固，全部保留现有接口）

### 层 1：tick 读取加固（解决 "0 readable ticks"）
1. **条带裁剪增强**：y 条带宽度加缓冲（`y_axis_col+5` → `y_axis_col+2*label_h`），
   上边界 `y0-20` → `y0-label_h`；x 条带高度同样按 `label_h` 自适应；
2. **旋转文本处理**：y 条带额外做 ±90° 旋转识别（或开启 PaddleOCR
   `use_textline_orientation`），覆盖论文图竖排 y 刻度标签（根因 #6）；
3. **多尺度 OCR + 投票**：条带分别以 1x/2x/4x 识别，同框文本按
   `rec_score × 长度` 投票；数字类文本优先；
4. **OCR 前预处理**：CLAHE 对比度归一化（深底/低对比图）、可选去噪；
5. **整图 OCR 兜底**：条带识别结果为空时，降级为整图 OCR + 几何过滤
   （按"轴线外侧"区域过滤文本框）——覆盖条带裁剪偏差的根因 #1；
6. **解析容错**：`parse_number_text` 增加脏文本清洗（`x`/`X` 乘号、
   半角全角空格、`O`→`0` 仅在数字上下文、`．` 全角点、上下标 Unicode 全集）；
   记录 parse 失败样本到 warning（可诊断）；
7. **映射校验（RANSAC）**：fit_axis 前对 (pixel, value) 做线性/对数 RANSAC，
   剔除离群刻度（旋转/误关联导致），提高少刻度场景成功率；
8. **轴端点锚点 fallback**：刻度 <2 但 OCR 有文本时，用"标签中心 + 文本值"
   直接建 tick（已有）；再不行，用轴端点像素 + 单个刻度值定标（借鉴
   WebPlotDigitizer 两点校准）→ 阈值从 ≥2 放宽为 ≥1+端点（在
   `coordinate_mapper` 加 `min_ticks=1` + 端点模式）。

### 层 2：标题 / 轴标题 / 单位识别（新能力）
1. **文本区域检测**：
   - 短期（纯 CV + 几何）：顶部带 = 标题区（plot_bbox 上方 0-15% 图高）；
     底部带 = x 轴标题区（x 轴线下方、刻度标签之下）；
     左侧带 = y 轴标题区（y 轴线左侧、刻度标签之外）；
   - 中期：YOLO `axis_title` 检测框（Phase B.5 训练后）给出精确区域，
     与几何带互为兜底；
2. **文本识别**：对区域做 PaddleOCR（复用层 1 的多尺度/预处理）；
3. **角色分类**（规则优先，数据积累后可训分类器）：
   - 位置规则：顶部→title；底部→x_label；左侧→y_label；
   - 内容规则：含 `(%)/(MPa)/(mm)` 括号单位或 `/` 单位 → 轴标题；
     以 `log`/`ln` 开头或含 `log` → 对数先验；
   - 输出结构化：`{role, text, variable, unit, log_hint}`；
4. **接入下游**：
   - `fit_axis` 的 `kind_hint`：轴标题含 log 先验 → 直接给 log 提示
     （解决"刻度少时 R² 判别不稳"）；
   - 结果 JSON/CSV 元数据：输出标题、轴变量、单位（竞赛答辩与真实使用价值）；
5. **评估指标**：新增"文本识别准确率"（tick 文本识别率、title/轴标题/单位
   解析正确率）——在合成/平台集上统计（GT 侧车里有轴标签、标题文本）。

### 层 3：坐标类型判别增强（目标 ≥98%）
- 现判据：R² 双拟合（≥3 刻度时可靠；2 刻度时任何模型都完美拟合 → 线性胜出，
  有 warning）；**真实 OCR 基线证明该判据在刻度误读时失效（56/98 判错）**；
- 增强（三信号融合投票）：
  1. **刻度值序列校验**：读到的刻度值必须是等差（线性）或等比（对数）序列
     （容差内）；不满足 → 判定存在误读，触发重试/RANSAC 剔除/整图 OCR 兜底；
  2. **像素间距分析**：对数轴主刻度像素间距近似相等（等数量级），线性轴
     等值间距相等——独立于 OCR 值的几何证据；
  3. **轴标题先验**（层 2）：标题含 log/ln → log 先验；
  - 融合：几何间距（不依赖 OCR 值）+ 序列校验（值）+ 标题先验（文本），
    三票中 ≥2 票即定；R² 拟合仅作数值映射参数（不再承担类型判别重任）。

### 层 4：YOLOv8-nano 结构检测（联动，Phase B.5）
- 用已有 `data/train_platform/*_yolo.txt`（6 类，含 tick_label/axis_title）
  训练检测器；检测框给文本区域精确定位（替代/增强纯几何条带），
  输出仍为同一 `ChartStructure`；文本识别仍走 PaddleOCR（检测与识别解耦，
  符合 PaddleOCR 范式）。

## 五、实施顺序与验证

| 步骤 | 内容 | 验证 |
|------|------|------|
| B-0 | 收集失败样例（用户提供失败图 ≥3 张）+ 建 debug 诊断包 | 定位"0 readable ticks"根因 |
| B-1 | 层 1 tick 加固（条带/多尺度/兜底/解析/RANSAC/端点锚点） | 失败样例全部恢复；合成 40 + 平台 100（stub + paddle）回归不回退 |
| B-2 | 层 2 标题/轴标题/单位识别 + 结构化输出 | 合成/平台集 title/单位解析率 ≥90%（GT 对照） |
| B-3 | 层 3 坐标判别三信号融合 | 轴类型准确率 ≥98%（合成+平台集，含 2 刻度场景） |
| B-4 | 层 4 YOLOv8 检测（与 B-3 可并行） | mAP@0.5 达标后接入，对照 ablation |
| B-5 | 真实图验证（用户已收集的 10-15 张） | 人工抽查 + gold 标注 RMSE |

每步保持 `pytest 60/60` + evaluate 回归（基准：合成 med ≤0.40%、平台 ≤1% 达标率 100%）。

## 六、待办/依赖用户

1. **失败样例图**（最重要的输入）：用户在测试中遇到 "y-axis: only 0
   readable ticks" 的 1-3 张原图 → 放 `data/real_papers/raw/`；
2. 真实论文图收集（Phase A.3 继续，≥10-15 张）；
3. 方案确认后按 B-0 → B-5 实施。

## 参考文献

- ChartEye（arXiv:2408.16123 / IEEE 10410945）：https://ieeexplore.ieee.org/document/10410945
- ICDAR 2019 CHART-Infographics 竞赛：https://ieeexplore.ieee.org/abstract/document/8978105
- DeMatch（ICDAR 2021）：https://dl.acm.org/doi/10.1007/978-3-030-86334-0_45
- Generalization of Fine Granular Extractions（ICDAR 2023）：https://dlnext.acm.org/doi/10.1007/978-3-031-41679-8_6
- PlotQA 数据集：https://github.com/NiteshMethani/PlotQA
- WebPlotDigitizer 坐标系统文档：https://deepwiki.com/automeris-io/WebPlotDigitizer/4.3-coordinate-systems-and-axes
- Engauge Digitizer：https://github.com/akhuettel/engauge-digitizer
