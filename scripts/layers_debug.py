"""Debug: run each layer detector on before.jpeg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
from PIL import Image

from pipeline.layers import (_detect_banners, _detect_strips,
                             _detect_corner_logo, _detect_cutouts)

img = np.array(Image.open("tests/samples/before.jpeg").convert("RGB"))
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

print("banners:", [c.box for c in _detect_banners(img, hsv)])
print("strips:", [(c.box, round(c.meta["edge_density"], 3))
                  for c in _detect_strips(img, hsv, gray)])
print("logos:", [c.box for c in _detect_corner_logo(img, hsv)])
try:
    print("cutouts:", [c.box for c in _detect_cutouts(img, gray)])
except Exception as e:
    print("cutouts ERR:", type(e).__name__, e)
