#!/bin/bash
# E1m: gentle right-end hard-example (fix E1 over-correction)
# E1 (hard3: fade 0.7/sigma~2.1 + weight 2.0) fixed 16 previously-failed
# curves but broke 27 (concentrated on clean real_mainstream/mcc/three_stage).
# E1m uses GENTLE data (hard3g: fade 0.35, sigma 1.0-2.2, crossing 0.6)
# + weight 1.3 -- keeps the right-end signal without biasing clean images.
cd /root/baseline
export PATH=/root/miniconda3/bin:$PATH
nohup /root/miniconda3/bin/python -u train/train_segmentation_multi.py \
  --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single,data/train_platform_hard3g \
  --val-dir data/val_multi,data/val_single \
  --epochs 25 --batch 16 --size 512 --per-dir-limit 600 \
  --init models/checkpoints/unet_multi_chain.pt \
  --out models/checkpoints/unet_multi_e1m.pt \
  --ema-decay 0.999 \
  --zone-contrast 1.0 --zone-dist 4 --zone-margin 0.3 --goi-ortho 0.5 \
  --chain-weight 0.15 --right-weight 1.3 \
  > /root/e1m_train.log 2>&1 &
echo "PID: $!"
