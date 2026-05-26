# V0 设计说明（OpenCV Baseline）

统一入口：Web `/extract` 与 API `/api/extract` 均调用 `app.pipeline.run_extraction`。

Pipeline：
1. 读取原图并保存 input.png
2. 按 `plot_area` 裁剪并保存 cropped_plot_area.png
3. OpenCV 提取 `curve_mask.png`（gray/hsv）
4. 连通域降噪 + 列中位数重建像素点（全局坐标）
5. `pixel_to_data_points` 映射为真实 x/y
6. 导出 CSV/JSON/overlay/redrawn/report

限制：仅面向清晰单曲线 V0，不含 OCR/深度学习/自动检测。
