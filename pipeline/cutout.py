"""SAM2 box-prompted alpha cutouts (anchors, logo strips, badges)."""
import logging
import os

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
        # SAM2 runs on CPU by default: on this machine, once YOLO/LaMa have
        # used the GPU, later CUDA convolutions silently corrupt SAM2's
        # masks into half-body slivers (verified A/B; matches the known
        # fp16-NaN fault of this GPU). CPU output is bit-identical quality
        # at ~2-4s/image. Opt back into CUDA with SAM2_DEVICE=cuda.
        device = os.environ.get("SAM2_DEVICE", "cpu")
        model = build_sam2_hf(SAM2_ID, device=device)
        # NO max_hole_area/max_sprinkle_area here: those activate sam2's
        # optional post-processing, which hard-crashes without the compiled
        # _C extension — and the crash path hands back MANGLED masks.
        _predictor = SAM2ImagePredictor(model, mask_threshold=0.0)
    return _predictor


def box_alpha(img_rgb: np.ndarray, box: tuple, feather: int = 3) -> np.ndarray:
    """SAM2 mask for `box`=(x,y,w,h) -> uint8 alpha (0-255), feathered.

    multimask_output=True is mandatory here: single-hypothesis mode
    intermittently locks onto a limb-sized sliver (cmp1's floating-face bug).
    """
    import torch
    x, y, w, h = box
    pred = _get_predictor()
    with torch.inference_mode():
        pred.set_image(img_rgb)
        masks, scores, _ = pred.predict(
            box=np.array([x, y, x + w, y + h], dtype=np.float32),
            multimask_output=True)
    m = masks[int(np.argmax(scores))]
    return _mask_to_alpha(m, box, feather)


def box_alpha_multi(img_rgb: np.ndarray, box: tuple,
                    feather: int = 3) -> tuple:
    """(best-scored alpha, union-of-all-hypotheses alpha).

    Segmentation sources disagree under soft-matted / low-contrast subjects;
    the caller tries candidates in order instead of trusting one verdict.
    """
    import torch
    x, y, w, h = box
    pred = _get_predictor()
    with torch.inference_mode():
        pred.set_image(img_rgb)
        masks, scores, _ = pred.predict(
            box=np.array([x, y, x + w, y + h], dtype=np.float32),
            multimask_output=True)
    best = masks[int(np.argmax(scores))]
    acc = None
    for m in masks:
        f = cv2.resize(m[y:y + h, x:x + w].astype(np.float32), (w, h))
        acc = f if acc is None else acc + f
    union_a = ((acc > 0.3).astype(np.uint8)) * 255
    union_a = cv2.GaussianBlur(union_a, ((feather | 1), (feather | 1)), 0)
    return _mask_to_alpha(best, box, feather), union_a


def _mask_to_alpha(m, box, feather):
    """masks arrive at FULL image resolution — crop to the box FIRST,
    then resize. Resizing the uncropped frame squeezes the subject into a
    corner of the card (the long-standing 'tiny anchor' bug)."""
    x, y, w, h = [int(v) for v in box]
    f = cv2.resize(m[y:y + h, x:x + w].astype(np.float32), (w, h))
    a = (f > 0).astype(np.uint8) * 255
    k = feather | 1
    a = cv2.GaussianBlur(a, (k, k), 0)
    a[a > 160] = 255
    return a


def box_alpha_grabcut(img_rgb: np.ndarray, box: tuple) -> np.ndarray:
    """Model-free foreground extraction via GrabCut seeded by layout priors:
    border ring = background, central core = probable subject. Rescues busy
    backdrops where every color/segmentation model degenerates (astro art).
    """
    x, y, w, h = [int(v) for v in box]
    crop = img_rgb[y:y + h, x:x + w]
    gc = np.zeros((h, w), np.uint8)
    gc[:] = cv2.GC_BGD
    m = int(0.08 * min(h, w))
    gc[m:h - m, m:w - m] = cv2.GC_PR_BGD
    ch_, cw_ = h // 2, w // 2
    gc[max(0, ch_ - h // 4):min(h, ch_ + h // 4),
       max(0, cw_ - w // 5):min(w, cw_ + w // 5)] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), gc, None,
                bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    a = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                 255, 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 128).astype(np.uint8), 8)
    if n > 1:
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        a[lab != j] = 0
    a = cv2.erode(a, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(a, (5, 5), 0)


def box_alpha_fill(img_rgb: np.ndarray, box: tuple, tol: int = 42) -> np.ndarray:
    """Geometric last-resort matting, no models involved.

    Pasted subject zones sit over near-flat chroma by layout design: sample
    border colors, drop anything within tolerance AND locally flat, keep the
    dominant subject component. Works on white, gray and astro backdrops;
    fails gracefully toward an almost-solid mask on fully busy borders.
    """
    x, y, w, h = [int(v) for v in box]
    crop = img_rgb[y:y + h, x:x + w].astype(np.int16)
    border = np.concatenate([
        crop[:8].reshape(-1, 3), crop[-8:].reshape(-1, 3),
        crop[:, :8].reshape(-1, 3), crop[:, -8:].reshape(-1, 3)])
    q = (border // 24).astype(np.int32)
    _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    refs = []
    for k in np.argsort(-cnt)[:5]:
        if cnt[k] < max(60, 0.04 * len(border)):
            break
        refs.append(border[inv == k].mean(axis=0))
    if not refs:                       # busy-border guard: single median
        refs.append(np.median(border, axis=0))
    ref = np.stack(refs)
    d = np.linalg.norm(crop[..., None, :] - ref[None], axis=-1).min(axis=-1)
    gray = cv2.cvtColor(img_rgb[y:y + h, x:x + w], cv2.COLOR_RGB2GRAY)
    f = gray.astype(np.float32)
    loc_std = np.sqrt(np.maximum(cv2.blur(f * f, (9, 9)) -
                                 cv2.blur(f, (9, 9)) ** 2, 0))
    bg = (d < tol) & (loc_std < 26)
    a = np.where(bg, 0, 255).astype(np.uint8)
    # open FIRST so loose props (bag straps, mic booms) can never bridge
    # into the subject blob during component analysis, then close gaps
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 128).astype(np.uint8), 8)
    if n > 1:                          # keep the dominant subject blob only
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = (lab == j) | (lab == 0)
        a[~keep] = 0
        a[lab == j] = 255
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    a = cv2.erode(a, np.ones((5, 5), np.uint8))   # shave matting-halo rim
    a = cv2.GaussianBlur(a, (5, 5), 0)
    return a
