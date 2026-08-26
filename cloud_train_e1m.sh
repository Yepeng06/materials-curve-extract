#!/bin/bash
# E1m: gentle right-end hard-example (fix E1 over-correction)
# E1 (right_fade 0.7 data + right-weight 2.0) fixed 16 previously-failed
# curves but broke 27 previously-perfect ones. E1m halves the pixel
# weight (2.0 -> 1.3) keeping the same hard3 data; isolates whether the
# damage came from the weight or the data itself.
cd /root/baseline
export PATH=/root/miniconda3/bin:$PATH
nohup /root/miniconda3/bin/python -u train/train_segmentation_multi.py \
  --data-dir data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single,data/train_platform_hard3 \
  --val-dir data/val_multi,data/val_single \
  --epochs 25 --batch 16 --size 512 --per-dir-limit 600 \
  --init models/checkpoints/unet_multi_chain.pt \
  --out models/checkpoints/unet_multi_e1m.pt \
  --ema-decay 0.999 \
  --zone-contrast 1.0 --zone-dist 4 --zone-margin 0.3 --goi-ortho 0.5 \
  --chain-weight 0.15 --right-weight 1.3 \
  > /root/e1m_train.log 2>&1 &
echo "PID: $!"
