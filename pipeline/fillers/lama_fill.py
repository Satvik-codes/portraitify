"""FAST tier filler: Big-LaMa (TorchScript) reusing the pre-cached weights."""
import logging
import numpy as np

from config import config

log = logging.getLogger("h2v.lama")
ENGINE = "big-lama"


def available() -> bool:
    return config.LAMA_PT_PATH.exists()


def _make_simple_lama(device_arg):
    from simple_lama_inpainting import SimpleLama
    if device_arg is None:
        return SimpleLama()
    try:
        return SimpleLama(device=device_arg)
    except TypeError:  # older versions pick cuda automatically
        return SimpleLama()


def _infer(model, image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    from PIL import Image
    out = model(Image.fromarray(image_rgb), Image.fromarray(mask_u8))
    return np.array(out.convert("RGB"))


def fill(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """image_rgb HxWx3 uint8 full canvas, mask_u8 255=fill. Returns filled RGB."""
    import torch
    import cv2
    from pipeline.vrac import model_slot

    dev = config.device()

    try:
        with model_slot(ENGINE, lambda: _make_simple_lama(dev if dev == "cuda" else None)) as model:
            out = _infer(model, image_rgb, mask_u8)
            # SimpleLama pads to /8 internally and returns the padded canvas
            return out[:image_rgb.shape[0], :image_rgb.shape[1]]

    except torch.cuda.OutOfMemoryError:
        log.warning("LaMa OOM on GPU - single CPU retry at long-side<=1280")
        scale = min(1.0, 1280.0 / max(image_rgb.shape[:2]))
        small = (cv2.resize(image_rgb, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else image_rgb)
        msmall = (cv2.resize(mask_u8, (small.shape[1], small.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
                  if scale < 1.0 else mask_u8)

        with model_slot("lama-cpu", lambda: _make_simple_lama("cpu")) as model:
            filled_small = _infer(model, small, msmall)
            filled_small = filled_small[:small.shape[0], :small.shape[1]]

        back = (cv2.resize(filled_small, (image_rgb.shape[1], image_rgb.shape[0]),
                           interpolation=cv2.INTER_CUBIC)
                if scale < 1.0 else filled_small)
        keep = mask_u8 < 128  # CPU roundtrip must never touch protected pixels
        out = back.copy()
        out[keep] = image_rgb[keep]
        return out
