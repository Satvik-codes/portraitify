"""Instrument _detect_strips checks on before.jpeg."""
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

m = ((hsv[..., 2] > config.LAYER_STRIP_LIGHT_V) &
     (hsv[..., 1] < config.LAYER_STRIP_LIGHT_S)).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
m = cv2.erode(m, np.ones((5, 5), np.uint8))
edges = cv2.Canny(gray, 80, 160)

n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
print(f"components: {n-1}")
for i in range(1, n):
    x, y, w, h, area = stats[i]
    if area < 0.004 * H * W or w < 0.12 * W - 10 or h < 0.025 * H:
        continue
    x, y, w, h = max(0, x - 3), max(0, y - 3), w + 6, h + 6
    cy = (y + h / 2) / H
    fill = m[y:y + h, x:x + w].mean() / 255.0
    density = float((edges[y:y + h, x:x + w] > 0).mean())
    ok_cy = cy >= 0.55
    ok_fill = fill >= 0.55
    ok_den = density >= config.LAYER_STRIP_EDGE_DENSITY
    print(f"cand ({x},{y},{w},{h}) cy={cy:.2f}({'Y' if ok_cy else 'N'}) "
          f"fill={fill:.2f}({'Y' if ok_fill else 'N'}) "
          f"density={density:.3f}({'Y' if ok_den else 'N'})")
