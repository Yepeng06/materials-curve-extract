#!/bin/bash
# E3: lever1+3 joint hyperparameter grid -- variant A: stronger contrast + chain
# Data: base 4 dirs + hard3 (same mix as E1), chain weight 0.22, zone 1.4
cd /root/baseline
export PATH=/root/miniconda3/bin:$PATH
nohup /root/miniconda3/bin/python -u train/train_segmentation_multi.py \
  --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single,data/train_platform_hard3 \
  --val-dir data/val_multi,data/val_single \
  --epochs 25 --batch 16 --size 512 --per-dir-limit 600 \
  --init models/checkpoints/unet_multi_chain.pt \
  --out models/checkpoints/unet_multi_e3a.pt \
  --ema-decay 0.999 \
  --zone-contrast 1.4 --zone-dist 4 --zone-margin 0.5 --goi-ortho 0.5 \
  --chain-weight 0.22 --right-weight 1.0 \
  > /root/e3a_train.log 2>&1 &
echo "PID: $!"
