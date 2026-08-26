"""Graphic-layer detection for news-graphic screenshots.

Decomposes an image into LayerCards (banner / strip / corner_logo / cutout)
plus the remaining photo layer. Deterministic thresholds tuned against
tests/samples/before.jpeg; see implementation-plan-v3.md section 4.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import cv2

from config import config

log = logging.getLogger("h2v.layers")


@dataclass
class LayerCard:
    kind: str            # banner | strip | corner_logo | cutout
    box: tuple           # (x, y, w, h) in source coords
    crop: np.ndarray     # source pixels (RGB)
    alpha: np.ndarray    # HxW uint8, 255 opaque
    z: int
    meta: dict = field(default_factory=dict)


@dataclass
class LayerPlan:
    cards: list
    graphic_union: np.ndarray
    found: bool

    @property
    def photo_mask(self):
        return cv2.bitwise_not(self.graphic_union)


def _components(mask, min_area, min_w, min_h):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and w >= min_w and h >= min_h:
            out.append((int(x), int(y), int(w), int(h)))
    return out


def _feather_alpha(w, h, feather):
    a = np.full((h, w), 255, np.uint8)
    if feather > 0:
        k = feather | 1
        a = cv2.GaussianBlur(a, (k, k), 0)
        a[feather:h - feather, feather:w - feather] = 255
    return a


def _detect_banners(img, hsv):
    H, W = img.shape[:2]
    m = ((hsv[..., 1] > config.LAYER_BANNER_SAT) &
         (hsv[..., 2] > config.LAYER_BANNER_VAL)).astype(np.uint8) * 255
    # Projection-profile rectangle extraction: finds the dense red row-band
    # (the banner) even when clothing bridges connect it to other red blobs.
    rows = m.mean(axis=1) / 255.0
    idx = (rows > 0.30).astype(np.uint8) * 255
    idx = cv2.morphologyEx(idx, cv2.MORPH_CLOSE,
                           np.ones((25,), np.uint8))  # bridge text-dip rows
    run, best_run, best_end = 0, 0, 0
    for i, v in enumerate(idx):
        run = run + 1 if v else 0
        if run > best_run:
            best_run, best_end = run, i
    if best_run < 0.03 * H:
        return []
    y0 = best_end - best_run + 1
    y1 = best_end + 1
    cols = m[y0:y1].mean(axis=0) / 255.0
    run, best_run, best_end = 0, 0, 0
    for i, v in enumerate(cols):
        run = run + 1 if v > 0.30 else 0
        if run > best_run:
            best_run, best_end = run, i
    if best_run < 0.15 * W:
        return []
    x0 = best_end - best_run + 1
    x1 = best_end + 1
    # extend both edges over the banner's lower-density tail (text zones)
    while x1 < W and m[y0:y1, x1].mean() / 255.0 > 0.12:
        x1 += 1
    while x0 > 0 and m[y0:y1, x0 - 1].mean() / 255.0 > 0.12:
        x0 -= 1
    x, y, w, h = x0, y0, x1 - x0, y1 - y0
    sub = m[y:y + h, x:x + w]
    if sub.mean() / 255.0 < 0.5:
        return []
    sat = hsv[y:y + h, x:x + w, 1]
    hue = hsv[y:y + h, x:x + w, 0].astype(np.float32)[sat > 100]
    if hue.size < 100:
        return []
    # Circular-hue concentration (red wraps 0/179; std is meaningless here).
    med = float(np.median(hue))
    d = np.abs(hue - med)
    d = np.minimum(d, 179 - d)
    if float((d <= 15).mean()) < 0.85:
        return []
    crop = img[y:y + h, x:x + w]
    return [LayerCard(
        "banner", (x, y, w, h), crop, _feather_alpha(w, h, 2), z=40,
        meta={"rel_w": w / W, "rel_cx": (x + w / 2) / W,
              "rel_cy": (y + h / 2) / H})]


def _detect_strips(img, hsv, gray):
    H, W = img.shape[:2]
    m = ((hsv[..., 2] > config.LAYER_STRIP_LIGHT_V) &
         (hsv[..., 1] < config.LAYER_STRIP_LIGHT_S)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.erode(m, np.ones((5, 5), np.uint8))  # sever bridges to bright crowd
    edges = cv2.Canny(gray, 80, 160)
    cards = []
    best = None
    for (x, y, w, h) in _components(m, 0.004 * H * W, 0.12 * W - 10, 0.025 * H):
        x, y, w, h = max(0, x - 3), max(0, y - 3), w + 6, h + 6  # grow back
        if (y + h / 2) / H < 0.55:
            continue  # captions live low in news graphics
        if m[y:y + h, x:x + w].mean() / 255.0 < 0.55:
            continue
        density = float((edges[y:y + h, x:x + w] > 0).mean())
        if density < config.LAYER_STRIP_EDGE_DENSITY:
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)
    for (x, y, w, h) in ([best] if best else []):
        crop = img[y:y + h, x:x + w]
        cards.append(LayerCard(
            "strip", (x, y, w, h), crop, _feather_alpha(w, h, 2), z=30,
            meta={"rel_w": w / W, "rel_cx": (x + w / 2) / W,
                  "rel_cy": (y + h / 2) / H, "edge_density": density}))
    return cards


def _detect_corner_logo(img, hsv):
    H, W = img.shape[:2]
    m = (hsv[..., 1] > 120).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lo, hi = config.LAYER_CORNER_LOGO_AREA
    cards = []
    best = None
    for (x, y, w, h) in _components(m, lo * H * W, 0.03 * W, 0.03 * H):
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        near_corner = (cx < 0.18 or cx > 0.82) and (cy < 0.15 or cy > 0.85)
        area_frac = (w * h) / (H * W)
        if not near_corner or not (lo <= area_frac <= hi):
            continue
        if best is None or w * h > best[2] * best[3]:
            best = (x, y, w, h)
    for (x, y, w, h) in ([best] if best else []):
        crop = img[y:y + h, x:x + w]
        cards.append(LayerCard(
            "corner_logo", (x, y, w, h), crop, _feather_alpha(w, h, 2), z=20,
            meta={"rel_w": w / W, "rel_cx": cx, "rel_cy": cy}))
    return cards


def _ring_segments_uniform(gray, x, y, w, h, ring=10, thresh=None):
    """Per-segment ring std: dict top/bottom/left/right -> std."""
    H, W = gray.shape[:2]
    thresh = thresh if thresh is not None else config.LAYER_CUTOUT_RING_STD
    x0, y0 = max(0, x - ring), max(0, y - ring)
    x1, y1 = min(W, x + w + ring), min(H, y + h + ring)
    sub = gray[y0:y1, x0:x1].astype(np.float32)
    outer = np.full(sub.shape[:2], 255, np.uint8)
    outer[(y - y0):(y - y0 + h), (x - x0):(x - x0 + w)] = 0
    segs = {}
    for name, sl in (("top", np.s_[:ring, :]), ("bottom", np.s_[-ring:, :]),
                     ("left", np.s_[:, :ring]), ("right", np.s_[:, -ring:])):
        vals = sub[sl][outer[sl] == 255]
        segs[name] = float(vals.std()) if vals.size else 255.0
    return segs


def _detect_cutouts(img, gray):
    H, W = img.shape[:2]
    try:
        from pipeline.detect import _load_yolo
        model = _load_yolo()
        dev = 0 if config.device() == "cuda" else "cpu"
        results = model.predict(img, conf=0.5, classes=[0], device=dev,
                                verbose=False)
    except Exception as e:
        log.warning("cutout YOLO failed: %s", e)
        return []
    cards = []
    for r in results:
        if r.boxes is None or not len(r.boxes):
            continue
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = x2 - x1, y2 - y1
            if w < 0.12 * W or h < 0.3 * H:
                continue
            segs = _ring_segments_uniform(gray, x1, y1, w, h)
            uniform = sum(1 for v in segs.values() if v < config.LAYER_CUTOUT_RING_STD)
            # News anchors stand at the frame edge over clean backgrounds;
            # edge-touch + dominance is the robust signal (rings are noisy).
            touches = (x1 <= 24) or (x2 >= W - 24) or (y2 >= H - 24)
            is_anchor = touches and w >= 0.25 * W and h >= 0.5 * H
            if uniform < 2 and not is_anchor:
                continue
            alpha = _feather_alpha(w, h, 3)
            if r.masks is not None and i < len(r.masks.data):
                m = r.masks.data[i].cpu().numpy()
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                am = (m * 255).astype(np.uint8)
                am = cv2.GaussianBlur(am, (7, 7), 0)
                am[am > 140] = 255
                alpha = np.minimum(alpha, am)
            crop = img[y1:y2, x1:x2]
            cards.append(LayerCard(
                "cutout", (x1, y1, w, h), crop, alpha, z=10,
                meta={"rel_w": w / W, "rel_cx": (x1 + w / 2) / W,
                      "rel_cy": (y1 + h / 2) / H,
                      "uniform_segs": uniform}))
    return cards


def _resolve_overlaps(cards):
    """Priority banner > strip > corner_logo > cutout; drop smaller overlaps."""
    prio = {"banner": 0, "strip": 1, "corner_logo": 2, "cutout": 3}
    keep = []
    for c in sorted(cards, key=lambda c: prio[c.kind]):
        bx, by, bw, bh = c.box
        clash = False
        for k in keep:
            kx, ky, kw, kh = k.box
            ix = max(0, min(bx + bw, kx + kw) - max(bx, kx))
            iy = max(0, min(by + bh, ky + kh) - max(by, ky))
            if ix * iy > 0.3 * min(bw * bh, kw * kh):
                clash = True
                break
        if not clash:
            keep.append(c)
    return keep


def detect_layers(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cards = (_detect_banners(img, hsv) + _detect_strips(img, hsv, gray) +
             _detect_corner_logo(img, hsv) + _detect_cutouts(img, gray))
    cards = _resolve_overlaps(cards)
    union = np.zeros((H, W), np.uint8)
    for c in cards:
        if c.kind == "cutout":
            continue  # anchor stays part of the photo layer (after.png ref)
        x, y, w, h = c.box
        union[y:y + h, x:x + w] = 255
    log.info("layers: %s", [(c.kind, c.box) for c in cards])
    return LayerPlan(cards=cards, graphic_union=union, found=bool(cards))


def clean_photo(img, plan):
    """Inpaint graphic holes out of the photo (LaMa full-res; MI-GAN fallback).

    LaMa first: large removed regions (banners/strips) need structure-aware
    filling at full resolution — MI-GAN's fixed 512px pass leaves ghosts.
    """
    if not plan.found:
        return img
    from pipeline.fillers import lama_fill
    try:
        filled = lama_fill.fill(img, plan.graphic_union)
        engine = "big-lama"
    except Exception as e:
        log.warning("lama cleanup failed (%s) -> migan", e)
        from pipeline.fillers import migan_fill
        filled = migan_fill.fill(img, plan.graphic_union)
        engine = "migan"
    keep = plan.graphic_union < 128
    filled[keep] = img[keep]
    log.info("photo cleaned via %s", engine)
    return filled


if __name__ == "__main__":
    import sys
    from PIL import Image
    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    plan = detect_layers(img)
    vis = img.copy()
    for c in plan.cards:
        x, y, w, h = c.box
        color = {"banner": (0, 0, 255), "strip": (255, 0, 255),
                 "corner_logo": (255, 160, 0), "cutout": (0, 255, 0)}[c.kind]
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 4)
        cv2.putText(vis, c.kind, (x + 6, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    out = config.DEBUG_DIR / "layers_demo.jpg"
    Image.fromarray(vis).save(out, quality=92)
    print([(c.kind, c.box) for c in plan.cards])
    print("overlay ->", out)


# --- Composition planner ------------------------------------------------------

def plan_composition(plan, canvas_w, canvas_h, centroid_y):
    """Map detected cards onto the target canvas. Returns placed list of
    (card, x, y, w, h) with photo rect handled separately by decide_paste."""
    placed = []
    banner = next((c for c in plan.cards if c.kind == "banner"), None)
    strip = next((c for c in plan.cards if c.kind == "strip"), None)
    logo = next((c for c in plan.cards if c.kind == "corner_logo"), None)


    if banner is not None:
        w = int(canvas_w * 0.92)
        h = int(banner.crop.shape[0] * w / banner.crop.shape[1])
        x = (canvas_w - w) // 2
        y = canvas_h - h - 36
        placed.append((banner, x, y, w, h))
    if strip is not None:
        w = int(canvas_w * 0.70)
        h = int(strip.crop.shape[0] * w / strip.crop.shape[1])
        x = (canvas_w - w) // 2
        y = (canvas_h - h - 36 if banner is None else
             canvas_h - 36 - (banner.crop.shape[0] * int(canvas_w * 0.92) // banner.crop.shape[1]) - 24 - h)
        placed.append((strip, x, y, w, h))
    if logo is not None:
        w = int(min(0.20 * canvas_w, max(0.08 * canvas_w,
                                        logo.meta["rel_w"] * canvas_w)))
        h = int(logo.crop.shape[0] * w / logo.crop.shape[1])
        placed.append((logo, canvas_w - w - 24, 24, w, h))
    return placed