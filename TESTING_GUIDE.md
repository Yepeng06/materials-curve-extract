# 实操测试指南（照着做即可）

> 环境：Anaconda 环境 **`mci`**（已装好所有依赖），项目目录 `F:\CODE\New\baseline`。
> 全程约 10-20 分钟。每节都是独立可运行的。

## 0. 打开终端并准备环境

```powershell
conda activate mci
cd F:\CODE\New\baseline
```

> 若 `conda activate` 报错，先执行 `conda init powershell` 再重开终端。

---

## 1. 单图体验（最快看到效果，1 分钟）

```powershell
python scripts/run_baseline.py --image data/eval_platform/img_0000.png --out-dir data/outputs --ocr stub --segmenter unet
```

**看结果**（3 个文件，在 `data/outputs\` 下）：

| 文件 | 内容 |
|------|------|
| `img_0000_overlay.png` | **叠加图**：原图上画出提取的曲线（最重要，直接看效果） |
| `img_0000.csv` | 提取出的曲线数据（x,y 两列） |
| `img_0000.json` | 完整结构化结果（轴类型、刻度值、耗时、警告等） |

**对照"标准答案"**：`data/eval_platform/img_0000.csv` 是同名 GT 文件（生成时
记录的原始数据），可以和你提取的 csv 对比（用 Excel/WPS 打开，或画个散点图）。

**多试几张**：把 `img_0000` 换成 `img_0001`…`img_0099` 随便挑，感受不同
模板/风格（黑白论文、多曲线对比、低质量截图、网格干扰等）。

---

## 2. 批量评估（复现报告里的数字，2-3 分钟）

```powershell
# 平台测试集 100 张（预期：达标率 ~100%）
python scripts/evaluate.py --data-dir data/eval_platform --out-dir data/eval_myrun --ocr stub --segmenter unet

# 合成测试集 40 张（预期：中位 ~0.4%，达标率 ~87.5%）
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval_myrun2 --ocr stub --segmenter unet
```

**看结果**：
- 终端末尾的 summary 就是关键指标（rel_rmse 中位/最大、≤1% 达标率）；
- `data/eval_myrun\report.csv` 是每张图的明细（哪张图差、哪张好一目了然）。

> 数字和你机器上的报告值可能略有出入（几百分之一的波动正常），但量级应一致：
> 平台集达标率应 ≈100%，合成集 ≈80-90%。

---

## 3. 对比 CV 后端（无训练后备，感受差距）

```powershell
python scripts/evaluate.py --data-dir data/synthetic --out-dir data/eval_cv --ocr stub --segmenter cv
```

预期：中位 ~1.7%、达标率 ~40%，且有灾难样本（最大 ~80%）。
这是"不用深度学习"的旧方案，用来证明 U-Net 路线的必要性。

---

## 4. 真实 OCR（PaddleOCR，单图约 6 秒，CPU）

```powershell
python scripts/run_baseline.py --image data/eval_platform/img_0000.png --out-dir data/outputs --ocr paddle --segmenter unet
```

- **第一次运行会下载 OCR 模型权重**（几十 MB，耐心等）；
- `--ocr stub` 用的是"标准答案"刻度文本（确定性、快，用于测试）；
  `--ocr paddle` 是真实文字识别（模拟真实论文图，有识别噪声）；
- 速度慢是正常的：CPU 版 PaddleOCR ~6s/图，全流程瓶颈在 OCR 不在模型。

---

## 5. 自己生成新数据再测（数据工厂，2 分钟）

用 dataset-platform 适配器生成 10 张**全新的**单曲线图（含 GT），然后评估：

```powershell
python data/dataset_builder.py --out-dir data/mytest --count 10 --num-curves 1 --seed 42
python scripts/evaluate.py --data-dir data/mytest --out-dir data/eval_mytest --ocr stub --segmenter unet
```

也可以生成**多曲线**训练风格的图（不带单曲线 GT，纯看效果）：

```powershell
python data/dataset_builder.py --out-dir data/mytest_multi --count 6 --seed 43
python scripts/run_baseline.py --image-dir data/mytest_multi --out-dir data/outputs_multi --ocr stub --segmenter unet
```

> 老版自包含工厂也可用：`python scripts/gen_synthetic.py --out-dir data/mytest2 --count 10 --seed 42`
> （注意：`data/mytest2` 下的 `*_mask.png` 是掩码，不要直接当输入图测）。

---

## 6. 单元测试（1 分钟）

```powershell
python -m pytest tests -q
```

预期输出：`60 passed`。含端到端管线测试（会临时生成图、跑 U-Net 推理），
慢是正常的（约 30 秒）。

---

## 7. 常见问题排查

| 现象 | 原因/解决 |
|------|-----------|
| OCR 很慢/卡住 | 你用了 `--ocr paddle` 或默认 `auto`；快速测试请显式 `--ocr stub` |
| 提示 `unet checkpoint not found` | 缺 `models/checkpoints/unet_curve.pt`（30MB，已入库，`git pull` 后应有） |
| 提示找不到图片 | 先 `Get-ChildItem data` 确认目录存在；`data/` 不入库，换机器需重新生成 |
| 显存不足（OOM） | 512 推理单图 <1GB，一般不会；若出现请关掉其他占显存程序 |
| 评估数字与文档差一点 | 正常波动；平台集 100% 达标、合成集中位 ≤0.5% 是稳定结论 |
| `conda activate` 失败 | `conda init powershell` 后重开终端 |

---

## 8. 想深入看内部流程？

加 `--debug` 保存每个步骤的中间图（结构检测、OCR 框、掩码、骨架）：

```powershell
python scripts/run_baseline.py --image data/eval_platform/img_0005.png --out-dir data/outputs_dbg --ocr stub --segmenter unet --debug
# 中间图在 data/outputs_dbg\debug\ 下
```

## 9. 下一步（等你测完）

- 收集真实论文蠕变图 → 放 `data/real_papers\raw\`（指南见 `data/real_papers\README.md`），
  AI 用 `--ocr paddle` 帮你评估真实图精度；
- Phase B（YOLOv8 结构检测）随时可开工。
