"""Verify the new val path (3-tuple loader + dice_iou) with a real checkpoint."""
import sys, os, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import torch
import numpy as np
import train_segmentation_multi as T
from mci.models.segmentation.unet import UNet, dice_iou

ckpt_path = "models/checkpoints/unet_multi_curve_512c.pt"
model = UNet(in_channels=1, base=64, out_channels=6).eval()
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
print("loaded", ckpt_path, "val_iou in ckpt:", ckpt.get("val_iou"))

cache = {}; skel_cache = {}
pairs = T._list_pairs("data/val_single")[:8]
ds = T.ChartDataset(pairs, augment=False, size=512, cache=cache, skel_cache=skel_cache)
print("skel_cache entries after first load:", len(skel_cache))
t0 = time.time()
v_dice = v_iou = 0.0
for i in range(len(ds)):
    x, y, s = ds[i]
    with torch.no_grad():
        prob = torch.sigmoid(model(x.unsqueeze(0)))
    d, iou = dice_iou(prob, y.unsqueeze(0))
    v_dice += d; v_iou += iou
    # skeleton sanity
    sk = s.numpy()
    print(f"  img{i}: dice={d:.4f} iou={iou:.4f} active_ch={int((y.sum(dim=(1,2))>0).sum().item())} skel_px={int(sk.sum())}")
print(f"avg dice={v_dice/len(ds):.4f} avg iou={v_iou/len(ds):.4f} ({time.time()-t0:.1f}s for {len(ds)} imgs incl cache build)")
