"""Element re-layout engine: rearrange the image's own regions into a
vertical composition (blurred backdrop + hero crop + banner/logo cards).

Deterministic, CPU-only, zero NaN risk. Used as the final fallback when
continuation engines produce voidy bands, and available as an explicit tier.
"""
import logging

import numpy as np
import cv2

from config import config

log = logging.getLogger("h2v.relayout")
ENGINE = "relayout"


def _find_banner(img_rgb: np.ndarray):
    """Largest saturated uniform-hue block in the BOTTOM half (captions)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    m = ((s > 140) & (v > 90)).astype(np.uint8)
    m[:img_rgb.shape[0] // 3] = 0  # banners/captions live low in the frame
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best, best_area = None, 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area and area > 0.01 * img_rgb.shape[0] * img_rgb.shape[1]:
            best_area, best = area, i
    if best is None:
        return None
    x, y, w, h = stats[best, 0], stats[best, 1], stats[best, 2], stats[best, 3]
    if h < 24 or w < 100:
        return None
    return (int(x), int(y), int(w), int(h))  # NOTE: x,y,w,h format


def _hero_box(img_rgb: np.ndarray):
    try:
        from ultralytics import YOLO
        from config import config as _c
        model = YOLO(str(_c.YOLO_PT_PATH))
        dev = 0 if _c.device() == "cuda" else "cpu"
        r = model.predict(img_rgb, conf=0.4, classes=[0], device=dev, verbose=False)
        if r and len(r[0].boxes):
            b = max(r[0].boxes.xyxy.cpu().numpy(),
                    key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]))
            return tuple(int(v) for v in b)
    except Exception as e:
        log.warning("hero detect failed: %s", e)
    H, W = img_rgb.shape[:2]
    return (W // 6, H // 8, W - W // 6, H * 3 // 5)


def _card(src, box, target_w, radius=28, pad=10):
    """Crop box from source, scale to width, rounded corners + soft shadow."""
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = src[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None, 0
    scale = target_w / crop.shape[1]
    card = cv2.resize(crop, (target_w, max(24, int(crop.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)
    ch = card.shape[0]
    canvas = np.zeros((ch + pad * 2, target_w + pad * 2, 4), np.uint8)
    mask = np.zeros(card.shape[:2], np.uint8)
    cv2.rectangle(mask, (0, 0), (target_w - 1, ch - 1), 255, -1)
    mask = _rounded(mask, radius)
    canvas[pad:pad + ch, pad:pad + target_w, :3] = card
    canvas[pad:pad + ch, pad:pad + target_w, 3] = mask
    return canvas, ch + pad * 2


def _rounded(mask, r):
    m = mask.copy()
    for cy, cx in [(r, r), (r, -r - 1), (-r - 1, r), (-r - 1, -r - 1)]:
        y0 = r if cy == r else mask.shape[0] + cy + 1
        x0 = r if cx == r else mask.shape[1] + cx + 1
        yy, xx = np.mgrid[0:2 * r, 0:2 * r]
        corner = (xx - r) ** 2 + (yy - r) ** 2 > r * r
        region = m[y0:y0 + 2 * r, x0:x0 + 2 * r]
        if region.shape[:2] == corner.shape:
            region[corner] = 0
    return m


def _paste_rgba(base, card, cx, cy):
    h, w = card.shape[:2]
    x0, y0 = cx - w // 2, cy - h // 2
    x0 = max(0, min(x0, base.shape[1] - w))
    y0 = max(0, min(y0, base.shape[0] - h))
    roi = base[y0:y0 + h, x0:x0 + w]
    a = card[..., 3:4].astype(np.float32) / 255.0
    base[y0:y0 + h, x0:x0 + w] = (card[..., :3].astype(np.float32) * a +
                                  roi.astype(np.float32) * (1 - a)).astype(np.uint8)


def compose(img_rgb: np.ndarray, ratio: str = "9:16") -> np.ndarray:
    cw, chh = config.CANVAS_SIZES[ratio]
    H, W = img_rgb.shape[:2]

    # backdrop: cover-crop + heavy blur + darken
    scale = max(cw / W, chh / H)
    bg = cv2.resize(img_rgb, (int(W * scale + 1), int(H * scale + 1)),
                    interpolation=cv2.INTER_AREA)
    bx, by = (bg.shape[1] - cw) // 2, (bg.shape[0] - chh) // 2
    bg = bg[by:by + chh, bx:bx + cw]
    bg = cv2.GaussianBlur(bg, (61, 61), 0)
    bg = (bg.astype(np.float32) * 0.72).astype(np.uint8)

    # hero
    x1, y1, x2, y2 = _hero_box(img_rgb)
    hero_w = int(cw * 0.86)
    hero, hh = _card(img_rgb, (x1, y1, x2, y2), hero_w)
    if hero is None:
        return bg
    _paste_rgba(bg, hero, cw // 2, int(chh * 0.34))

    # banner (caption belongs near the bottom) - convert x,y,w,h -> x1,y1,x2,y2
    banner_box = _find_banner(img_rgb)
    if banner_box:
        bx, by, bw2, bh2 = banner_box
        card, bh = _card(img_rgb, (bx, by, bx + bw2, by + bh2),
                         int(cw * 0.92), radius=20)
        if card is not None:
            _paste_rgba(bg, card, cw // 2, chh - bh // 2 - 40)

    # channel logo: source top-right corner -> canvas top-right
    lw = int(W * 0.16)
    logo_crop = img_rgb[0:int(H * 0.14), W - lw:W]
    if logo_crop.size and logo_crop.std() > 8:
        card, lh = _card(img_rgb, (W - lw, 0, W, int(H * 0.14)),
                         int(cw * 0.20), radius=14, pad=6)
        if card is not None:
            _paste_rgba(bg, card, cw - card.shape[1] // 2 - 24,
                        card.shape[0] // 2 + 24)

    log.info("relayout composed: hero=%s banner=%s", (x1, y1, x2, y2), banner_box)
    return bg
