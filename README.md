# 材料科学图像曲线识别与智能解析（V0 OpenCV Baseline）

快速跳转：[简介](#简介) | [首次使用](#首次使用) | [后续使用](#后续使用) | [如何使用系统](#如何使用系统)


# 简介
## 项目目标
构建 **Web 优先** 端到端基线：上传曲线图 + 手工输入坐标信息，输出结构化曲线数据。

## V0 当前能力
- FastAPI + Jinja2 Web 入口，`/extract` 与 `/api/extract` 共用 `run_extraction()`。
- 手工输入 `plot_area`、`x_range`、`y_range`。
- OpenCV 曲线提取：`gray`（深色曲线）/ `hsv`（彩色曲线）两种模式。
- 输出 `output.csv`、`output.json`、`curve_mask.png`、`extracted_overlay.png`、`redrawn_curve.png`、`report.md`。

## 当前限制
- 适合清晰单曲线与简单彩色单曲线。
- 暂不保证复杂多曲线/交叉/严重噪声。
- 不做自动 OCR、自动 plot_area 检测、深度学习模型。
- HSV 阈值需手工输入。
# 首次使用
**1、拉取项目代码**

找一个文件夹用于存放代码，根目录终端执行：
```bash
 git clone -b codex/fix-templateresponse-calls-for-fastapi https://github.com/Yepeng06/materials-curve-extract.git
```
（没装git的话可以手动下载代码压缩包解压）
如图：
![](https://jueshipa-pic.oss-cn-beijing.aliyuncs.com/blog/20260527010850700.png)

**2、创建并激活虚拟环境**

根目录下终端执行：
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
激活成功后，命令行前面会出现 `(.venv)`
升级pip：
```
python -m pip install --upgrade pip
```
效果如图：
![](https://jueshipa-pic.oss-cn-beijing.aliyuncs.com/blog/20260527011330114.png)

**3、安装依赖**

项目根目录终端执行：
```bash
pip install -r requirements.txt
```
网络慢可尝试换源：（我网络还可以没有测试这条命令）
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
检验是否安装成功：成功则输出 `deps ok` 
```bash
python -c "import fastapi, cv2, numpy, pandas, matplotlib, httpx; print('deps ok')"
```
效果如图：
![](https://jueshipa-pic.oss-cn-beijing.aliyuncs.com/blog/20260527011717764.png)

**4、运行测试**

项目根目录终端执行：
```bash
$env:PYTHONPATH="."
pytest -q
```
测试通过会显示：`9 passed`

**5、启动web系统**

在项目根目录、虚拟环境已激活的状态下执行：
```bash
uvicorn app.main:app --reload
```
出现类似信息说明启动成功：
```bash
Uvicorn running on http://127.0.0.1:8000
```
访问链接：127.0.0.1:8000

# 后续使用

项目根目录执行：
```bash
.\.venv\Scripts\activate
uvicorn app.main:app --reload
```
# 如何使用系统
**上传图片**

选择需要提取曲线的图片，例如：
```
creep_000001.png
```
**plot_area**
`plot_area` 是绘图区在原图中的像素位置：
```
left   = 绘图区左边界像素 x 坐标top    = 绘图区上边界像素 y 坐标right  = 绘图区右边界像素 x 坐标bottom = 绘图区下边界像素 y 坐标
```
注意：这里填的是**像素坐标**，不是曲线真实坐标。
建议 plot_area 尽量只包含曲线所在的绘图区，不要包含：
```
标题、图例、坐标轴标签、坐标轴外文字、大面积空白
```
如果曲线和坐标轴边框都是黑色，plot_area 可以稍微往绘图区内部缩一点，减少边框被误提取。

**x_range**

`x_range` 是横坐标真实数值范围。
例如横坐标是时间，范围为 0 到 1000 h：
```
x_min = 0x_max = 1000
```

**y_range**

`y_range` 是纵坐标真实数值范围。
例如纵坐标是蠕变应变，范围为 0 到 0.5：
```
y_min = 0y_max = 0.5
```

**mode**

当前支持：
```
gray：适合黑色或深色曲线hsv：适合彩色曲线，需要手动设置 HSV 阈值
```
一般先使用：
```
mode = gray
```

**resample_n**

`resample_n` 是输出点数。  
例如填：
```
512
```
表示最终重采样输出 512 个点。

---

未完待续。。。以下内容未经验证：
## Python 调用示例
```python
from pathlib import Path
from app.config import AxisRange, PlotArea, ExtractionConfig
from app.pipeline import run_extraction

cfg = ExtractionConfig(
    input_path=Path("demo.png"),
    output_dir=Path("outputs/demo_run"),
    plot_area=PlotArea(left=100, top=80, right=900, bottom=700),
    x_range=AxisRange(min=0, max=1000),
    y_range=AxisRange(min=0, max=10),
    mode="gray",
    resample_n=512,
)
result = run_extraction(cfg)
print(result["output_files"]["output_csv"])
```
