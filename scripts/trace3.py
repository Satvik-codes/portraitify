"""Trace banner projection v2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
from PIL import Image
from config import config

img = np.array(Image.open("tests/samples/before.jpeg").convert("RGB"))
H, W = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

m = ((hsv[..., 1] > config.LAYER_BANNER_SAT) &
     (hsv[..., 2] > config.LAYER_BANNER_VAL)).astype(np.uint8) * 255
rows = m.mean(axis=1) / 255.0
idx = (rows > 0.30).astype(np.uint8) * 255
idx = cv2.morphologyEx(idx, cv2.MORPH_CLOSE, np.ones((25,), np.uint8))
run, best_run, best_end = 0, 0, 0
for i, v in enumerate(idx):
    run = run + 1 if v else 0
    if run > best_run:
        best_run, best_end = run, i
print(f"row run: {best_run} (need {0.03*H:.0f}) end={best_end}")
if best_run:
    y0, y1 = best_end - best_run + 1, best_end + 1
    cols = m[y0:y1].mean(axis=0) / 255.0
    run, bc, be = 0, 0, 0
    for i, v in enumerate(cols):
        run = run + 1 if v > 0.30 else 0
        if run > bc:
            bc, be = run, i
    print(f"col run: {bc} (need {0.15*W:.0f}) end={be}")
    if bc:
        x, y, w, h = be - bc + 1, y0, bc, best_run
        sub = m[y:y + h, x:x + w]
        sat = hsv[y:y + h, x:x + w, 1]
        hue = hsv[y:y + h, x:x + w, 0].astype(np.float32)[sat > 100]
        print(f"rect=({x},{y},{w},{h}) fill={sub.mean()/255:.2f} "
              f"sat_pixels={hue.size} hue_std={hue.std():.1f}")

