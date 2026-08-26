"""Mirror-pad pre-fill + placeholder scrub.

mirror_pad seeds empty canvas bands with flipped copies of adjacent real
content so texture fillers refine structure instead of inventing from flat
gray. scrub_placeholder repairs any surviving exact-placeholder-gray patches
after filling without ever touching protected pixels.
"""
import logging

import numpy as np
import cv2

log = logging.getLogger("h2v.prefill")


def _tile_to(src: np.ndarray, target_len: int, axis: int) -> np.ndarray:
    """Repeat src along axis, flipping each repetition so edges stay continuous."""
    pieces = [src]
    total = src.shape[0] if axis == 0 else src.shape[1]
    cur = src
    while total < target_len:
        cur = cur[::-1] if axis == 0 else cur[:, ::-1]
        pieces.append(cur)
        total += cur.shape[0] if axis == 0 else cur.shape[1]
    out = np.concatenate(pieces, axis=axis)
    return out[:target_len] if axis == 0 else out[:, :target_len]


def _progressive_soft(band: np.ndarray, seam_edge: str) -> np.ndarray:
    """Blend sharp-at-seam -> blurred-far-from-seam.

    seam_edge in {'top','bottom','left','right'} names where the ORIGINAL
    content sits relative to this band.
    """
    h, w = band.shape[:2]
    k_far = max(5, (min(h, w) // 6) | 1)
    k_near = max(3, (min(h, w) // 20) | 1)
    soft = cv2.GaussianBlur(band, (k_far, k_far), 0)
    near = cv2.GaussianBlur(band, (k_near, k_near), 0)

    n = h if seam_edge in ("top", "bottom") else w
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)      # 0 at index 0
    if seam_edge == "top":
        alpha = t                                        # seam at last row
    elif seam_edge == "bottom":
        alpha = t[::-1]                                  # seam at first row
    elif seam_edge == "left":
        alpha = t
    else:                                                # 'right'
        alpha = t[::-1]

    a = (alpha[:, None] if seam_edge in ("top", "bottom")
         else alpha[None, :])[..., None]
    return (near.astype(np.float32) * (1 - a) +
            soft.astype(np.float32) * a).astype(np.uint8)


def mirror_pad(canvas: np.ndarray, paste_box: tuple) -> np.ndarray:
    """Fill open bands of `canvas` with mirrored content growing off the box."""
    H, W = canvas.shape[:2]
    x, y, w, h = paste_box
    out = canvas.copy()

    top_h = y
    if top_h > 0:
        strip = out[y:y + min(top_h, h), x:x + w]
        band = _tile_to(np.flipud(strip), top_h, axis=0)
        out[:top_h, x:x + w] = _progressive_soft(band, "bottom")

    bot_h = H - (y + h)
    if bot_h > 0:
        strip = out[y + h - min(bot_h, h):y + h, x:x + w]
        band = _tile_to(np.flipud(strip), bot_h, axis=0)
        out[y + h:, x:x + w] = _progressive_soft(band, "top")

    left_w = x
    if left_w > 0:
        strip = out[y:y + h, x:x + min(left_w, w)]
        band = _tile_to(np.fliplr(strip), left_w, axis=1)
        out[y:y + h, :left_w] = _progressive_soft(band, "right")

    right_w = W - (x + w)
    if right_w > 0:
        strip = out[y:y + h, x + w - min(right_w, w):x + w]
        band = _tile_to(np.fliplr(strip), right_w, axis=1)
        out[y:y + h, x + w:] = _progressive_soft(band, "left")

    # corners: blur-fill so colliding mirror seams don't form hard edges
    grayish = cv2.GaussianBlur(out, (31, 31), 0)
    for sy, ey, sx, ex in [(0, y, 0, x), (0, y, x + w, W),
                           (y + h, H, 0, x), (y + h, H, x + w, W)]:
        if ey > sy and ex > sx:
            out[sy:ey, sx:ex] = grayish[sy:ey, sx:ex]
    return out


_PLACEHOLDER = 127.0


def scrub_placeholder(filled: np.ndarray, paste_box: tuple,
                      tol: int = 2, block: int = 9) -> np.ndarray:
    """Replace block-uniform exact-placeholder-gray patches outside the box."""
    H, W = filled.shape[:2]
    x, y, w, h = paste_box

    near_exact = (np.abs(filled.astype(np.int16) - int(_PLACEHOLDER))
                  .max(axis=2) <= tol).astype(np.uint8)
    hits = cv2.erode(near_exact, np.ones((block, block), np.uint8)).astype(bool)
    hits[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = False

    n = int(hits.sum())
    if n == 0:
        return filled

    mask = cv2.dilate(hits.astype(np.uint8), np.ones((15, 15), np.uint8)) * 255
    repaired = cv2.inpaint(filled, mask, 7, cv2.INPAINT_TELEA)
    # keep repair strictly outside the box (belt & braces)
    repaired[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = \
        filled[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
    log.info("scrubbed %d placeholder px", n)
    return repaired
