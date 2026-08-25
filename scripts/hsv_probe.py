"""Measure HSV stats in ground-truth regions of before.jpeg."""
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("tests/samples/before.jpeg").convert("RGB"))
H, W = img.shape[:2]
print("image:", W, "x", H)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def probe(name, x, y, w, h, cond):
    sub = hsv[y:y + h, x:x + w]
    frac = float(cond(sub).mean())
    print(f"{name}: pass={frac:.2f} | S med {int(np.median(sub[..., 1]))} "
          f"| V med {int(np.median(sub[..., 2]))} | H med {int(np.median(sub[..., 0]))}")


# ground-truth guesses (fractions of 1200x675)
probe("banner(red)", 500, 490, 650, 160,
      lambda s: (s[..., 1] > 140) & (s[..., 2] > 90))
probe("strip(white)", 30, 545, 380, 100,
      lambda s: (s[..., 2] > 200) & (s[..., 1] < 60))
probe("cornerlogo", 1060, 20, 120, 100,
      lambda s: s[..., 1] > 120)
