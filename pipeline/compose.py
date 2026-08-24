"""Final composition: paste original byte-identically, soften only background."""
import numpy as np
import cv2


def composite(filled_canvas: np.ndarray, scaled_original: np.ndarray,
              paste_box: tuple, ramp_px: int | None = None) -> np.ndarray:
    """Return canvas with `scaled_original` (box-sized HxWx3) pasted at paste_box.

    Box region: bytes are EXACTLY scaled_original (guarantee).
    Background-side feathering: a ramp_px-wide ring strictly OUTSIDE the box
    is blended between the filled background and a blurred copy of itself so
    upscale/seam mismatch never meets the paste edge as a hard cliff.
    """
    H, W = filled_canvas.shape[:2]
    x, y, w, h = paste_box
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)

    if scaled_original.shape[:2] != (h, w):
        raise ValueError(f"scaled_original {scaled_original.shape[:2]} != box {(h, w)}")

    out = filled_canvas.copy()

    if ramp_px is None:
        from config import config
        ramp_px = config.RAMP_PX

    if ramp_px > 0:
        k = max(3, ramp_px * 2 | 1)
        blurred = cv2.GaussianBlur(filled_canvas, (k, k), 0)
        box_mask = np.zeros((H, W), dtype=np.uint8)
        box_mask[y:y + h, x:x + w] = 1
        dist_out = cv2.distanceTransform(1 - box_mask, cv2.DIST_L2, 3)
        ring = (dist_out > 0) & (dist_out <= ramp_px)
        wt = (1.0 - dist_out[ring] / float(ramp_px)).astype(np.float32)[..., None]
        out[ring] = (wt * filled_canvas[ring].astype(np.float32) +
                     (1.0 - wt) * blurred[ring].astype(np.float32)).astype(np.uint8)

    out[y:y + h, x:x + w] = scaled_original
    return out


def assert_region_identical(result: np.ndarray, scaled_original: np.ndarray,
                            paste_box: tuple) -> None:
    """Hard invariant used by tests and pipeline self-check."""
    x, y, w, h = paste_box
    a = result[y:y + h, x:x + w]
    if not np.array_equal(a, scaled_original):
        diff = int(np.abs(a.astype(int) - scaled_original.astype(int)).sum())
        raise AssertionError(f"pixel guarantee violated in paste region, absdiff={diff}")
