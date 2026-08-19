"""Diagnose why smoke-trained model reports val_dice/val_iou = 0."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import numpy as np
from mci.models.segmentation.unet import UNet, dice_iou

ckpt = torch.load("data/_smoke_multi.pt", map_location="cpu", weights_only=False)
print("ckpt keys:", list(ckpt.keys()))
print("val_iou in ckpt:", ckpt.get("val_iou"), "epoch:", ckpt.get("epoch"), "ema:", ckpt.get("ema_decay"))
model = UNet(in_channels=1, base=64, out_channels=6)
model.load_state_dict(ckpt["state_dict"])
model.eval()
# random input -> check output distribution
x = torch.randn(1, 1, 512, 512)
with torch.no_grad():
    logit = model(x)
    prob = torch.sigmoid(logit)
print("output shape:", tuple(logit.shape))
print("prob min/mean/max:", float(prob.min()), float(prob.mean()), float(prob.max()))
print("logit mean abs:", float(logit.abs().mean()))

# load one real val image and run through the real pipeline
import glob, cv2
from mci.pipeline.chart_structure import detect_structure
from mci.pipeline.coordinate_mapper import build_axes
from mci.pipeline.extractor import load_config
from mci.pipeline.tick_reader import StubOCRBackend, read_ticks
from mci.utils import read_image
cfg = load_config()
for p in sorted(glob.glob("data/val_single/*.png"))[:2]:
    stem = os.path.splitext(p)[0]
    img = read_image(p)
    h, w = img.shape[:2]
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (512, 512), interpolation=cv2.INTER_AREA)
    xt = torch.from_numpy(small).float().unsqueeze(0).unsqueeze(0) / 255.0
    with torch.no_grad():
        logit = model(xt)
        prob = torch.sigmoid(logit)[0].numpy()
    print(os.path.basename(p), "prob max per channel:", [round(float(prob[c].max()), 3) for c in range(6)])
