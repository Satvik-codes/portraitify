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
VOID_FRACTION_MAX = 0.22      # >22% uniform band area = voidy
EDGE_DENSITY_MIN = 2.6        # mean |Laplacian| in bands below this = voidy


def band_metrics(img_rgb: np.ndarray, paste_box: tuple) -> dict:
    H, W = img_rgb.shape[:2]
    x, y, w, h = paste_box
    parts = []
    if y > 8:
        parts.append(img_rgb[:y - 4])
    if y + h < H - 8:
        parts.append(img_rgb[y + h + 4:])
    if x > 8:
        parts.append(img_rgb[:, :x - 4])
    if x + w < W - 8:
        parts.append(img_rgb[:, x + w + 4:])
    if not parts:
        return {"void_fraction": 0.0, "edge_density": 99.0, "voidy": False,
                "band_pixels": 0}

    norm = [cv2.resize(p, (W, min(p.shape[0], 2048))) for p in parts]
    band = np.concatenate(norm, axis=0)

    gray = np.ascontiguousarray(
        np.dot(band[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8))
    hh, ww = gray.shape
    blocks = gray[:hh // BLOCK * BLOCK, :ww // BLOCK * BLOCK] \
        .reshape(hh // BLOCK, BLOCK, ww // BLOCK, BLOCK)
    block_std = blocks.std(axis=(1, 3))
    void_fraction = float((block_std < VOID_BLOCK_STD).mean())

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edge_density = float(np.abs(lap).mean())

    # Colorful soft content (bokeh-style continuation) is a legitimate look;
    # dead bands are monochrome (black NaN, placeholder gray, flat smear).
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    mean_sat = float(hsv[..., 1].mean()) / 255.0
    near_monochrome = mean_sat < 0.06

    voidy = (void_fraction > VOID_FRACTION_MAX and near_monochrome) \
        or edge_density < 1.2
    return {"void_fraction": round(void_fraction, 4),
            "edge_density": round(edge_density, 2),
            "mean_saturation": round(mean_sat, 3),
            "voidy": bool(voidy), "band_pixels": int(hh * ww)}


def assert_not_void(img_rgb: np.ndarray, paste_box: tuple, engine: str) -> dict:
    m = band_metrics(img_rgb, paste_box)
    if m["voidy"]:
        log.warning("VOID signature from engine=%s: %s", engine, m)
    return m
