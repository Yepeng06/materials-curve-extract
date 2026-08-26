#!/bin/bash
# E3a: pure hyperparameter tuning WITHOUT hard3 data (isolate data vs loss)
# Data: base 4 dirs only (same as original chain). zone 1.4 / margin 0.5 / chain 0.22
cd /root/baseline
export PATH=/root/miniconda3/bin:$PATH
nohup /root/miniconda3/bin/python -u train/train_segmentation_multi.py \
  --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single \
  --val-dir data/val_multi,data/val_single \
  --epochs 25 --batch 16 --size 512 --per-dir-limit 600 \
  --init models/checkpoints/unet_multi_chain.pt \
  --out models/checkpoints/unet_multi_e3a.pt \
  --ema-decay 0.999 \
  --zone-contrast 1.4 --zone-dist 4 --zone-margin 0.5 --goi-ortho 0.5 \
  --chain-weight 0.22 --right-weight 1.0 \
  > /root/e3a_train.log 2>&1 &
echo "PID: $!"
