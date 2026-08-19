"""Smoke-test the new ChartDataset axes path (Phase C fix)."""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import train_segmentation_multi as T

pairs = T._list_pairs("data/val_multi")[:3]
print("pairs:", [os.path.basename(p) for p, _ in pairs])
cache = {}
ds = T.ChartDataset(pairs, augment=False, size=512, cache=cache)
x, y = ds[0]
print("x:", tuple(x.shape), "y:", tuple(y.shape))
print("channels active:", [int(y[c].sum().item()) for c in range(6)])
print("cache entries:", len(cache))
