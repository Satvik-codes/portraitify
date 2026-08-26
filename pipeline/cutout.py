"""SAM2 box-prompted alpha cutouts (anchors, logo strips, badges)."""
import logging

import numpy as np
import cv2

from config import config

log = logging.getLogger("h2v.cutout")
SAM2_ID = "facebook/sam2.1-hiera-base-plus"

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2_hf(SAM2_ID, device=config.device())
        _predictor = SAM2ImagePredictor(model, mask_threshold=0.0,
                                        max_hole_area=8.0, max_sprinkle_area=0.0)
    return _predictor


def box_alpha(img_rgb: np.ndarray, box: tuple, feather: int = 3) -> np.ndarray:
    """SAM2 mask for `box`=(x,y,w,h) -> uint8 alpha (0-255), feathered."""
    import torch
    x, y, w, h = box
    pred = _get_predictor()
    with torch.inference_mode():
        pred.set_image(img_rgb)
        masks, scores, _ = pred.predict(
            box=np.array([x, y, x + w, y + h], dtype=np.float32),
            multimask_output=False)
    m = masks[0]
    m = cv2.resize(m.astype(np.float32), (w, h))
    a = (m > 0).astype(np.uint8) * 255
    k = feather | 1
    a = cv2.GaussianBlur(a, (k, k), 0)
    a[a > 160] = 255
    return a
