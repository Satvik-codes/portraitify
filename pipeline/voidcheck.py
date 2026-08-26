"""Void detection: hard gate so no result can ship with dead bands.

A void is a large region with no meaningful content: block-uniform color
(placeholder gray, NaN black, featureless smear). We score the filled bands
on (a) fraction of uniform blocks and (b) edge density, and compare against
thresholds learned from real failures on this machine.
"""
import logging

import numpy as np
import cv2

log = logging.getLogger("h2v.voidcheck")

BLOCK = 16
VOID_BLOCK_STD = 2.5          # block std below this = uniform = void candidate
VOID_FRACTION_MAX = 0.30      # >30% uniform band area = voidy, color or not
EDGE_DENSITY_MIN = 2.6        # mean |Laplacian| in bands below this = voidy


def band_metrics(img_rgb: np.ndarray, paste_box: tuple, exclude_boxes=None) -> dict:
    H, W = img_rgb.shape[:2]
    x, y, w, h = paste_box
    parts = []       # (pixels, valid_mask) pairs in canvas space
    valid_full = np.ones((H, W), bool)
    if exclude_boxes:
        for (ex, ey, ew, eh) in exclude_boxes:
            valid_full[max(0, ey):ey + eh, max(0, ex):ex + ew] = False

    def take(a, m):
        return a, m

    if y > 8:
        parts.append(take(img_rgb[:y - 4], valid_full[:y - 4]))
    if y + h < H - 8:
        parts.append(take(img_rgb[y + h + 4:], valid_full[y + h + 4:]))
    if x > 8:
        parts.append(take(img_rgb[:, :x - 4], valid_full[:, :x - 4]))
    if x + w < W - 8:
        parts.append(take(img_rgb[:, x + w + 4:], valid_full[:, x + w + 4:]))
    if not parts:
        return {"void_fraction": 0.0, "edge_density": 99.0, "voidy": False,
                "band_pixels": 0}

    band = np.concatenate(
        [cv2.resize(p, (W, min(p.shape[0], 2048))) for p, _ in parts], axis=0)
    vmask = np.concatenate(
        [cv2.resize(m.astype(np.float32), (W, min(m.shape[0], 2048)),
                    interpolation=cv2.INTER_NEAREST) > 0.5
         for _, m in parts], axis=0)

    gray = np.ascontiguousarray(
        np.dot(band[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8))
    hh, ww = gray.shape
    bh, bw = hh // BLOCK, ww // BLOCK
    if bh == 0 or bw == 0:
        return {"void_fraction": 0.0, "edge_density": 99.0, "voidy": False,
                "band_pixels": int(hh * ww)}
    blocks = gray[:bh * BLOCK, :bw * BLOCK] \
        .reshape(bh, BLOCK, bw, BLOCK)
    vmask_b = vmask[:bh * BLOCK, :bw * BLOCK] \
        .reshape(bh, BLOCK, bw, BLOCK)
    valid_frac = vmask_b.mean(axis=(1, 3))
    block_std = blocks.std(axis=(1, 3))
    considered = valid_frac > 0.6          # blocks mostly outside cards
    void_blocks = (block_std < VOID_BLOCK_STD) & considered
    void_fraction = float(void_blocks.sum() / max(1, considered.sum()))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_valid = np.abs(lap)[vmask]
    edge_density = float(lap_valid.mean()) if lap_valid.size else 0.0

    # Strict gate: uniform bands fail regardless of color. Colored mud is mud.
    voidy = (void_fraction > VOID_FRACTION_MAX) or (edge_density < EDGE_DENSITY_MIN)
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    return {"void_fraction": round(void_fraction, 4),
            "edge_density": round(edge_density, 2),
            "mean_saturation": round(float(hsv[..., 1].mean()) / 255.0, 3),
            "voidy": bool(voidy), "band_pixels": int(vmask.sum())}


def assert_not_void(img_rgb: np.ndarray, paste_box: tuple, engine: str) -> dict:
    m = band_metrics(img_rgb, paste_box)
    if m["voidy"]:
        log.warning("VOID signature from engine=%s: %s", engine, m)
    return m
