"""Probe MI-GAN traced model input format."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from config import config

jit = torch.jit.load(str(config.TORCH_HOME / "hub" / "checkpoints" / "migan_traced.pt"),
                     map_location="cpu").to(config.device()).eval()

img = torch.rand(1, 3, 288, 512, device=config.device())
mask = (torch.rand(1, 1, 288, 512, device=config.device()) > 0.5).float()

candidates = {
    "3ch masked img": img * (1 - mask),
    "4ch img+mask": torch.cat([img, mask], dim=1),
    "4ch mask+img": torch.cat([mask, img], dim=1),
    "8ch img,mask,img,mask": torch.cat([img, mask, img, mask], dim=1),
}
for name, x in candidates.items():
    try:
        with torch.inference_mode():
            out = jit(x)
        print(f"{name}: OK -> out shape {tuple(out.shape)}")
    except Exception as e:
        print(f"{name}: {type(e).__name__} {str(e)[:90]}")
