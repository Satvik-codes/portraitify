"""Repro: after.png -> 2:3 fast tier, full traceback."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

import pipeline

img = np.array(Image.open("tests/samples/after.png").convert("RGB"))
print("input:", img.shape)
try:
    meta = pipeline.run(img, ratio="2:3", tier="fast", job_id="repro-23")
    print("OK", meta["paste_box"], meta["void"])
except Exception:
    traceback.print_exc()
