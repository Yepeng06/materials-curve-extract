import sys
sys.path.insert(0, r"F:\CODE\New\baseline\src")
import numpy as np
import cv2
from mci.pipeline.panel_detect import detect_panels

def render(rects, w=600, h=400, line_w=3):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for (x0, y0, x1, y1) in rects:
        img[y0:y1+1, x0:x0+line_w] = 0
        img[y0:y1+1, x1-line_w+1:x1+1] = 0
        img[y0:y0+line_w, x0:x1+1] = 0
        img[y1-line_w+1:y1+1, x0:x1+1] = 0
    return img

img = render([(60, 50, 540, 350)])
print("panels:", detect_panels(img))

# debug internals
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
cnts, _ = cv2.findContours(bw2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("n contours:", len(cnts))
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.03*peri, True)
    x, y, ww, hh = cv2.boundingRect(approx)
    print(f"  peri={peri:.0f} approx={len(approx)} bbox=({x},{y},{ww},{hh}) area={ww*hh}")
