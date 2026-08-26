#!/bin/bash
cd /root/baseline
export PATH=/root/miniconda3/bin:$PATH
python - <<'EOF'
import sys
sys.path.insert(0, 'src')
from train.train_segmentation_multi import _list_pairs, ChartDataset
pairs = _list_pairs('data/train_platform,data/train_platform_4c,data/train_platform_5c,data/train_platform_single,data/train_platform_hard3')
print('pairs:', len(pairs))
ds = ChartDataset(pairs[:8], augment=False, size=512, cache={}, skel_cache={}, chain_cache={}, chainx_cache={})
x, y, s, c, cx = ds[0]
print('sample shapes:', tuple(x.shape), tuple(y.shape), tuple(s.shape), 'chain:', None if c is None else tuple(c.shape))
import torch
assert tuple(x.shape) == (1, 512, 512)
print('dataset OK')
EOF
