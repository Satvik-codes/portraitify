"""Trace banner projection + cutout ring decisions."""
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
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# --- banner projection ---
m = ((hsv[..., 1] > config.LAYER_BANNER_SAT) &
     (hsv[..., 2] > config.LAYER_BANNER_VAL)).astype(np.uint8) * 255
rows = m.mean(axis=1) / 255.0
top5 = np.argsort(rows)[-5:]
print("densest rows:", sorted(top5.tolist()), "densities:",
      [round(float(rows[i]), 2) for i in sorted(top5.tolist())])
run, best_run, best_end = 0, 0, 0
for i, v in enumerate(rows):
    run = run + 1 if v > 0.35 else 0
    if run > best_run:
        best_run, best_end = run, i
print(f"best row run: {best_run} rows ending y={best_end} (need >= {0.03*H:.0f})")
if best_run:
    y0, y1 = best_end - best_run + 1, best_end + 1
    cols = m[y0:y1].mean(axis=0) / 255.0
    run, best_run_c, best_end_c = 0, 0, 0
    for i, v in enumerate(cols):
        run = run + 1 if v > 0.5 else 0
        if run > best_run_c:
            best_run_c, best_end_c = run, i
    print(f"best col run: {best_run_c} (need >= {0.15*W:.0f}) ending x={best_end_c}")
    if best_run_c:
        x0 = best_end_c - best_run_c + 1
        sub = m[y0:y1, x0:best_end_c + 1]
        hue = hsv[y0:y1, x0:best_end_c + 1, 0].astype(np.float32)
        print(f"rect=({x0},{y0},{best_run_c},{best_run}) fill={sub.mean()/255:.2f} "
              f"hue_std={hue.std():.1f}")

# --- cutout rings ---
from ultralytics import YOLO
model = YOLO(str(config.YOLO_PT_PATH))
r = model.predict(img, conf=0.5, classes=[0], device=0, verbose=False)[0]
for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 0.12 * W or h < 0.3 * H:
        continue
    ring = 10
    x0, y0 = max(0, x1 - ring), max(0, y1 - ring)
    xa, ya = min(W, x2 + ring), min(H, y2 + ring)
    sub = gray[y0:ya, x0:xa].astype(np.float32)
    outer = np.full(sub.shape, 255, np.uint8)
    outer[(y1 - y0):(y1 - y0 + h), (x1 - x0):(x1 - x0 + w)] = 0
    for name, sl in (("top", np.s_[:ring, :]), ("bottom", np.s_[-ring:, :]),
                     ("left", np.s_[:, :ring]), ("right", np.s_[:, -ring:])):
        vals = sub[sl][outer[sl] == 255]
        std = float(vals.std()) if vals.size else -1
        print(f"person({x1},{y1}) {name}: std={std:.1f} "
              f"({'uniform' if std < config.LAYER_CUTOUT_RING_STD else 'busy'})")
