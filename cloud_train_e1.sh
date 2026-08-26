#!/bin/bash
# E1: right-end hard-example training (right_fade data + right-end pixel weight)
# Data: base 4 dirs (per-dir 600) + hard3 (right_fade 62%) -- hard ratio ~20%
cd /root/baseline
export PATH=/root/miniconda3/bin:$PATH
nohup /root/miniconda3/bin/python -u train/train_segmentation_multi.py \
  --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single,data/train_platform_hard3 \
  --val-dir data/val_multi,data/val_single \
  --epochs 25 --batch 16 --size 512 --per-dir-limit 600 \
  --init models/checkpoints/unet_multi_chain.pt \
  --out models/checkpoints/unet_multi_e1.pt \
  --ema-decay 0.999 \
  --zone-contrast 1.0 --zone-dist 4 --zone-margin 0.3 --goi-ortho 0.5 \
  --chain-weight 0.15 --right-weight 2.0 \
  > /root/e1_train.log 2>&1 &
echo "PID: $!"
