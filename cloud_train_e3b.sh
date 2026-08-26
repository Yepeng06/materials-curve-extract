#!/bin/bash
# E3 variant B: lighter chain weight + longer approach zones (zone-dist 6)
# Tests whether chain-weight 0.10 with wider approach-zone contrast helps
cd /root/baseline
export PATH=/root/miniconda3/bin:$PATH
nohup /root/miniconda3/bin/python -u train/train_segmentation_multi.py \
  --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single,data/train_platform_hard3 \
  --val-dir data/val_multi,data/val_single \
  --epochs 25 --batch 16 --size 512 --per-dir-limit 600 \
  --init models/checkpoints/unet_multi_chain.pt \
  --out models/checkpoints/unet_multi_e3b.pt \
  --ema-decay 0.999 \
  --zone-contrast 1.0 --zone-dist 6 --zone-margin 0.3 --goi-ortho 0.5 \
  --chain-weight 0.10 --right-weight 1.0 \
  > /root/e3b_train.log 2>&1 &
echo "PID: $!"
