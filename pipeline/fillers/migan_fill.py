"""MI-GAN filler: fast texture-preserving inpaint (traced jit, ~26 MB).

Primary photo-cleanup engine for the smart path; LaMa stays the fallback.
Input contract (mirrors IOPaint's MiGAN usage): image normalized to [-1,1],
mask {0,1} with 1 = fill, canvas padded to /8 multiples.
"""
import logging

import numpy as np

from config import config

log = logging.getLogger("h2v.migan")
ENGINE = "migan"

_jit = None


def _model():
    global _jit
    if _jit is None:
        import torch
        _jit = torch.jit.load(str(config.TORCH_HOME / "hub" / "checkpoints" / "migan_traced.pt"),
                              map_location="cpu")
    return _jit


def fill(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """image HxWx3 uint8, mask 255=fill. Returns filled RGB."""
    import cv2
    import torch
    from pipeline.vrac import model_slot

    H, W = image_rgb.shape[:2]
    scale = min(1.0, 512.0 / max(H, W))
    gh, gw = max(8, round(H * scale / 8) * 8), max(8, round(W * scale / 8) * 8)
    small = cv2.resize(image_rgb, (gw, gh), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask_u8, (gw, gh), interpolation=cv2.INTER_NEAREST)

    def _run(model):
        t_img = torch.from_numpy(small.astype(np.float32) / 127.5 - 1.0) \
            .permute(2, 0, 1)[None].to(config.device())
        t_mask = torch.from_numpy((msmall > 127).astype(np.float32)) \
            .permute(2 if msmall.ndim == 3 else 0, 1)[None, None].to(config.device())
        with torch.inference_mode():
            out = model(t_img, t_mask)
        out = out[0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
        out = ((out + 1) * 127.5).astype(np.uint8)
        return cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)

    with model_slot(ENGINE, _model) as model:
        filled = _run(model)

    keep = mask_u8 < 128  # never touch protected pixels
    filled[keep] = image_rgb[keep]
    return filled
