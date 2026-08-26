# 训练侧实验计划（云端 4090，2026-08-26 第二战场）

> 目标：val 6a 0.8481 → 0.95（Phase D 0.8641 → 0.95）。
> 现状：后处理已近饱和（P1/S1b/lowconf 组合 val 6a 0.8537）；训练侧是剩余主杠杆。

## 诊断结论（data/CHAIN_FAILURE_DIAGNOSIS.md，子代理完成）

- **>5% 桶 18 条**：右端曲线丢失 ≈50%（decelerating_creep 6、lqs 3、grid 2），
  真接近区切换 33%（6 条），ghost 1、整链发散 1。
- **2-5% 桶 34 条**：接近区家族 59%（20/34）仍是主体；lqs 整条/大段偏移 4-5 条。
- **1-2% 桶 29 条**：log 轴像素精度残留（bias_y 已压缩 69→29）。

## 实验设计（按预期收益排序）

### E1: 右端丢失 hard-example 增强（目标 +2~4pp，回收 >5% 的 ~8-10 条）
右端丢失机制：pred 链在右半段（x_frac 0.75-0.94）渐进偏离 GT，曲线平缓贴轴 +
重模糊（blur≤0.5、shot_blur）时模型概率峰丢失。训练信号缺右端低置信区覆盖。
方案：
1. `_apply_hard_crossing` 增加 "right_fade" 模式：曲线右端 20-35% 渲染后做
   局部低对比/模糊（模拟概率峰丢失），GT 不变（模型必须学会在低置信区保持）；
2. 或者纯数据加权：训练时对右端 x_frac>0.7 的 mask 区域给更高损失权重
   （bce_dice 加 right-tail 权重项，如 ×2.0）；
3. lqs 模板增强：shot_blur sigma 上限 1.0→1.5、resize 下限 0.65→0.55。

### E2: 接近区专门增强（目标 +1~3pp，回收 2-5% 的 ~20 条中部分）
现有 hard2 的 cross/hug 已覆盖，但 chain2 显示 hard 占比 36% 过度保守。
方案：
1. hard 数据占比控制：基础 4 目录（2000+600+300+600）+ hard2 取 15-20% 比例
   （per-dir-limit 控制），而非 chain2 的 36%；
2. hug 间距更密（2-4px 当量）变体 + 更长的平行段（占 x 跨度 40-60%）。

### E3: 杠杆1+3 联合调参网格（目标 +1-2pp，低成本）
在 chain 基础上微调参数组合（本地烟雾 2ep 已验证续训可行）：
- chain-weight: 0.10 / 0.15 / 0.22
- zone-contrast: 0.8 / 1.0 / 1.4
- zone-margin: 0.2 / 0.3 / 0.5
- goi-ortho: 0.3 / 0.5
推荐先跑 3 组合（chain-weight 0.22 + zone 1.4；chain 0.10 + margin 0.5；
chain 0.15 + ortho 0.3），全部从 unet_multi_chain.pt 续训 25ep。

### E4: 链级可微追踪（大工程，暂缓）
直接优化追踪后链误差。工程量 2-3 天，风险高，作为论文深挖方向备用。

## 云端执行清单
1. 本地 git commit（后处理 lowconf 已生效，val 6a 0.8537）；
2. 生成 E2 数据（hard3：接近区 + 右端 fade 混合，~2000 张）并上传；
3. 用户开卡 → autodl.py run 训练命令（batch 16）；
4. 下载 checkpoint → 本地 eval（val + Phase D）→ 对比。

## 验收口径提醒
合成集 6a 95% 文献中无先例（LineFormer UB-PMC 93.1/88.25；EpiCurveBench
最强 VLM 42.9%）。若训练侧回收 5-8pp 至 ~0.90-0.92，需与用户对齐最终口径
（合成独立集 vs 真实图、6a vs 6b、含曲线数惩罚与否）。
