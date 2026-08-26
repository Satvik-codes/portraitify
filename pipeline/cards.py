"""Card building/pasting shared by smart + relayout engines."""
import numpy as np
import cv2


def rounded_alpha(w, h, r):
    m = np.full((h, w), 255, np.uint8)
    for cy, cx in ((r, r), (r, w - r - 1), (h - r - 1, r), (h - r - 1, w - r - 1)):
        y0, x0 = max(0, cy - r), max(0, cx - r)
        yy, xx = np.mgrid[0:min(h, cy + r) - y0 or 1, 0:min(w, cx + r) - x0 or 1]
        reg = m[y0:y0 + yy.shape[0], x0:x0 + xx.shape[0]]
        corner = (xx + x0 - cx) ** 2 + (yy + y0 - cy) ** 2 > r * r
        reg[corner] = 0
    return m


def build_card(crop, target_w, radius=20, pad=8, feather=3):
    """Scale crop to target_w, rounded corners, returns RGBA card."""
    ch, cw = crop.shape[:2]
    th = max(24, int(round(ch * target_w / cw)))
    card = cv2.resize(crop, (target_w, th), interpolation=cv2.INTER_AREA)
    alpha = np.full((th, target_w), 255, np.uint8)
    alpha = cv2.GaussianBlur(alpha, (feather | 1, feather | 1), 0)
    alpha[radius:th - radius, radius:target_w - radius] = 255
    if radius > 0:
        alpha[:radius, :radius][
            (np.mgrid[0:radius, 0:radius][0] - radius) ** 2 +
            (np.mgrid[0:radius, 0:radius][1] - radius) ** 2 > radius * radius] = 0
        alpha[:radius, -radius:][
            (np.mgrid[0:radius, 0:radius][0] - radius) ** 2 +
            (np.mgrid[0:radius, 0:radius][1]) ** 2 > radius * radius] = 0
        alpha[-radius:, :radius][
            (np.mgrid[0:radius, 0:radius][0]) ** 2 +
            (np.mgrid[0:radius, 0:radius][1] - radius) ** 2 > radius * radius] = 0
        alpha[-radius:, -radius:][
            (np.mgrid[0:radius, 0:radius][0]) ** 2 +
            (np.mgrid[0:radius, 0:radius][1]) ** 2 > radius * radius] = 0
    rgba = np.zeros((th + pad * 2, target_w + pad * 2, 4), np.uint8)
    rgba[pad:pad + th, pad:pad + target_w, :3] = card
    rgba[pad:pad + th, pad:pad + target_w, 3] = alpha
    return rgba


def paste_card(base, card, cx, cy, shadow_alpha=0):
    """Paste RGBA card centered at (cx, cy), clipped to base. Returns footprint."""
    h, w = card.shape[:2]
    x0 = int(max(0, min(cx - w // 2, base.shape[1] - w)))
    y0 = int(max(0, min(cy - h // 2, base.shape[0] - h)))
    roi = base[y0:y0 + h, x0:x0 + w].astype(np.float32)
    if shadow_alpha > 0:
        sh = np.zeros((h, w), np.float32)
        am = cv2.GaussianBlur(card[..., 3].astype(np.float32), (21, 21), 0)
        sh[am > 0] = shadow_alpha / 100.0
        dark = (roi * (1 - (sh[..., None] * 0.6))).astype(np.uint8)
        base[y0:y0 + h, x0:x0 + w] = dark
    a = card[..., 3:4].astype(np.float32) / 255.0
    blend = (card[..., :3].astype(np.float32) * a +
             base[y0:y0 + h, x0:x0 + w].astype(np.float32) * (1 - a))
    base[y0:y0 + h, x0:x0 + w] = blend.astype(np.uint8)
    return (x0 + 8, y0 + 8, w - 16, h - 16)  # inner footprint (excl pad)


def paste_exact(base, crop_scaled, alpha, x, y, feather=3):
    """Paste crop_scaled at (x,y) with alpha; guarantees exact pixels where alpha==255."""
    h, w = crop_scaled.shape[:2]
    H, W = base.shape[:2]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    crop_scaled = crop_scaled[:h, :w]
    a = alpha[:h, :w]
    solid = a >= 250
    base[y:y + h, x:x + w][solid] = crop_scaled[solid]
    soft = (a > 0) & (a < 250)
    af = (a[soft].astype(np.float32) / 255.0)[..., None]
    base[y:y + h, x:x + w][soft] = (
        crop_scaled[soft].astype(np.float32) * af +
        base[y:y + h, x:x + w][soft].astype(np.float32) * (1 - af)).astype(np.uint8)
    return (x, y, w, h)
