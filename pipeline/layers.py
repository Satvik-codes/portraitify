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
    photo_rect: tuple | None = None   # dominant photo region (texture-based)
    subject_union: np.ndarray | None = None  # anchor zones erased from the band


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
    # Projection-profile rectangle extraction over EVERY qualifying row-run,
    # bottom-first: a red-clothed foreground subject can out-saturate the
    # actual plate higher in frame; picking the single longest run highjacks
    # the search onto his kurta while the banner below goes unclaimed.
    rows = m.mean(axis=1) / 255.0
    idx = (rows > 0.30).astype(np.uint8) * 255
    idx = cv2.morphologyEx(idx, cv2.MORPH_CLOSE,
                           np.ones((25,), np.uint8))  # bridge text-dip rows
    runs, run_end = [], -1
    for i, v in enumerate(list(idx) + [0]):
        if v:
            if run_end < 0:
                run_start, run_end = i, i
            else:
                run_end = i
        elif run_end >= 0:
            if run_end - run_start + 1 >= max(4, int(0.03 * H)):
                runs.append((run_start, run_end))
            run_end = -1
    for y0, y1 in sorted(runs, reverse=True):       # bottom-most first
        cols = m[y0:y1].mean(axis=0) / 255.0
        crun, cbest, cend = 0, 0, 0
        for i, v in enumerate(cols):
            crun = crun + 1 if v > 0.30 else 0
            if crun > cbest:
                cbest, cend = crun, i
        if cbest < 0.15 * W:
            continue
        x0, x1 = cend - cbest + 1, cend + 1
        # extend both edges over the plate's lower-density tail (text zones)
        while x1 < W and m[y0:y1, x1].mean() / 255.0 > 0.12:
            x1 += 1
        while x0 > 0 and m[y0:y1, x0 - 1].mean() / 255.0 > 0.12:
            x0 -= 1
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        sub = m[y:y + h, x:x + w]
        if sub.mean() / 255.0 < 0.5:
            continue
        sat = hsv[y:y + h, x:x + w, 1]
        hue = hsv[y:y + h, x:x + w, 0].astype(np.int64)[sat > 100]
        if hue.size < 100:
            continue
        # Peak circular hue-bin share: banners mix accent-colored glyphs
        # (yellow numerals, gold serif) into the plate red; a single-median
        # concentration test misclassifies them. Peak-bin share keeps the
        # coherence requirement while tolerating accent noise.
        hi = np.bincount(hue, minlength=180).astype(np.float32)
        ext_ = np.concatenate([hi, hi])
        win = np.convolve(ext_, np.ones(25, np.float32), "same")[:180]
        p = int(np.argmax(win))
        d = np.abs(hue - p)
        d = np.minimum(d, 179 - d)
        if float((d <= 12).mean()) < 0.62:
            continue
        return [LayerCard(
            "banner", (x, y, w, h), img[y:y + h, x:x + w],
            _feather_alpha(w, h, 2), z=40,
            meta={"rel_w": w / W, "rel_cx": (x + w / 2) / W,
                  "rel_cy": (y + h / 2) / H})]
    return []


def _detect_strips(img, hsv, gray):
    """Nameplate strips, both polarities: light plates (metal/white bg) AND
    dark plates (gold-on-black etc). Geometry gates do the classification
    work; color only selects the polarity mask."""
    H, W = img.shape[:2]
    edges = cv2.Canny(gray, 80, 160)

    def _scan(mask, min_edge):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))  # sever bridges
        best = None
        density = 0.0
        for (x, y, w, h) in _components(mask, 0.004 * H * W,
                                        0.12 * W - 10, 0.025 * H):
            x, y, w, h = max(0, x - 3), max(0, y - 3), w + 6, h + 6
            if (y + h / 2) / H < 0.55:
                continue  # captions live low in news graphics
            if mask[y:y + h, x:x + w].mean() / 255.0 < 0.55:
                continue
            density = float((edges[y:y + h, x:x + w] > 0).mean())
            if density < min_edge:
                continue
            if best is None or w * h > best[2] * best[3]:
                best = (x, y, w, h)
        return best, density

    # Edge floor applies to light metallic plates; sparse-serif dark plates
    # (thin gold strokes over black) fail ANY absolute density bar, so that
    # pass leans on the geometry+solidity gates alone.
    best, density = _scan(((hsv[..., 2] > config.LAYER_STRIP_LIGHT_V) &
                           (hsv[..., 1] < config.LAYER_STRIP_LIGHT_S))
                          .astype(np.uint8) * 255,
                          config.LAYER_STRIP_EDGE_DENSITY)
    best_d, density_d = _scan(((hsv[..., 2] < 80) &
                               (hsv[..., 1] < 90)).astype(np.uint8) * 255, 0.0)
    if best_d is not None and (best is None or
                               best_d[2] * best_d[3] > best[2] * best[3]):
        best, density = best_d, density_d
        log.info("strip matched as dark plate")
    cards = []
    for (x, y, w, h) in ([best] if best else []):
        crop = img[y:y + h, x:x + w]
        cards.append(LayerCard(
            "strip", (x, y, w, h), crop, _feather_alpha(w, h, 2), z=30,
            meta={"rel_w": w / W, "rel_cx": (x + w / 2) / W,
                  "rel_cy": (y + h / 2) / H, "edge_density": density}))
    return cards


def _detect_corner_logo(img, hsv):
    """Up to two corner badges by descending area; overlap resolution later
    sacrifices whichever collide with people/plates — emitting several keeps
    the real bug alive when skin-tone blobs out-area it."""
    H, W = img.shape[:2]
    m = (hsv[..., 1] > 120).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    lo, hi = config.LAYER_CORNER_LOGO_AREA
    cands = []
    for (x, y, w, h) in _components(m, lo * H * W, 0.03 * W, 0.03 * H):
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        near_corner = (cx < 0.18 or cx > 0.82) and (cy < 0.15 or cy > 0.85)
        area_frac = (w * h) / (H * W)
        if not near_corner or not (lo <= area_frac <= hi):
            continue
        # corner proximity is the channel-bug signature; raw area promotes
        # random skin/prop blobs that merely sit near edges
        dx = min(cx, 1 - cx)
        dy = min(cy, 1 - cy)
        cands.append(((dx * dx + dy * dy) ** 0.5, x, y, w, h))
    picked = []
    for (_d, x, y, w, h) in sorted(cands, key=lambda t: t[0]):
        dup = False
        for (px, py, pw, ph) in picked:
            ix = max(0, min(x + w, px + pw) - max(x, px))
            iy = max(0, min(y + h, py + ph) - max(y, py))
            if ix * iy > 0.3 * min(w * h, pw * ph):
                dup = True
                break
        if not dup and len(picked) < 2:
            picked.append((x, y, w, h))
    crop_cards = []
    for (x, y, w, h) in picked:
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        crop_cards.append(LayerCard(
            "corner_logo", (x, y, w, h), img[y:y + h, x:x + w],
            _feather_alpha(w, h, 2), z=20,
            meta={"rel_w": w / W, "rel_cx": cx, "rel_cy": cy}))
    return crop_cards


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


def _detect_photo_rect(img):
    """Dominant photo region via texture energy — backdrop-agnostic.

    Photos are gradient-dense; flat or graphics margins are not. Color is
    never consulted: white, gray and astro-graphic backdrops all fail the
    texture gate equally.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tex = np.abs(gx) + np.abs(gy)
    bs = 16
    hb, wb = H // bs, W // bs
    if hb < 4 or wb < 4:
        return None
    tb = tex[:hb * bs, :wb * bs].reshape(hb, bs, wb, bs).mean(axis=(1, 3))
    mask = (tb > 12.0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(stats[j, k]) * bs for k in range(4))
    if w < 0.35 * W or h < 0.35 * H:
        return None
    return (x, y, w, h)


def _backdrop_is_white(img, box, margin=14):
    """Share of near-white pixels in a ring just outside `box`.

    Median-of-ring fails when the anchor abuts a busy photo (median dilutes
    toward photo colors); a plain white-share is robust for the decision
    this feeds: may we key out near-white halos safely?
    """
    H, W = img.shape[:2]
    x, y, w, h = box
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
    ring = np.zeros((y1 - y0, x1 - x0), bool)
    ring[:y - y0] = True
    ring[y + h - y0:] = True
    ring[:, :x - x0] = True
    ring[:, x + w - x0:] = True
    vals = hsv[y0:y1, x0:x1][ring]
    if vals.size < 50:
        return False
    return float(((vals[:, 2] > 185) & (vals[:, 1] < 70)).mean()) > 0.45


def _detect_cutouts(img, gray, photo_rect=None, strip_box=None):
    """One foreground person (the anchor) via structure, not backdrop color.

    YOLO finds everyone; the anchor scores high by being tall with his head
    high in frame, touching a frame edge, and standing outside (or barely
    inside) the dominant photo rect — crowd people sit fully inside it and
    lose. At most one cutout is emitted: recomposing two foreground people
    from one source frame has no editorial meaning here.
    """
    H, W = img.shape[:2]
    try:
        from pipeline.detect import _load_yolo
        model = _load_yolo()
        dev = 0 if config.device() == "cuda" else "cpu"
        results = model.predict(img, conf=0.45, classes=[0], device=dev,
                                verbose=False)
    except Exception as e:
        log.warning("cutout YOLO failed: %s", e)
        return []
    best = None
    mask_best = None
    for r in results:
        if r.boxes is None or not len(r.boxes):
            continue
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            conf = float(r.boxes.conf[i])
            bx1, by1, bx2, by2 = [int(v) for v in box]
            bw_, bh_ = bx2 - bx1, by2 - by1
            # structural gates: prominent person, head high in frame
            if bw_ < 0.10 * W or bh_ < 0.40 * H or by1 > 0.25 * H:
                continue
            area = bw_ * bh_
            touches = (bx1 <= 24) or (bx2 >= W - 24) or (by2 >= H - 24)
            score = float(area) * (1.35 if touches else 1.0)
            if photo_rect is not None:
                px, py, pw, ph = photo_rect
                ix = max(0, min(bx2, px + pw) - max(bx1, px))
                iy = max(0, min(by2, py + ph) - max(by1, py))
                inside = (ix * iy) / max(1, area)
                # On textured backdrops (gray footage, astro art) the photo
                # rect swallows the anchor's own column too — so containment
                # only disqualifies near-total insiders (true crowd); anchor
                # likelihood otherwise decays smoothly with containment.
                if inside >= 0.96 and not touches:
                    continue          # fully scene-internal -> crowd
                score *= (1.0 - 0.7 * inside)
                if inside <= 0.35:
                    score *= 1.15     # beside/before the photo -> strong
            score *= (0.8 + 0.4 * conf)
            if best is None or score > best[0]:
                best = (score, (bx1, by1, bx2, by2), conf)
                mask_best = r.masks.data[i].cpu().numpy() \
                    if (r.masks is not None and i < len(r.masks.data)) else None
    if best is None:
        return []
    _, (x1, y1, x2_pre, y2_pre), _conf = best
    x2, y2 = x2_pre, y2_pre
    # Ground rules against a detected plate line:
    #   • candidate crosses the strip line -> he stands BEHIND the plate:
    #     trim his box to the plate top so plate pixels never enter SAM2.
    #   • candidate ends well above frame bottom on a lower-edge-running box
#     -> extend to his natural ground (plate top or frame bottom).
    trimmed = False
    limit = H
    if strip_box is not None:
        sx, sy, sw_, sh_ = strip_box
        ox = max(0, min(x2, sx + sw_) - max(x1, sx))
        if ox > 0.4 * (x2 - x1):
            if y1 < sy - 20 and y2 > sy + 20:
                y2 = sy
                trimmed = True
            elif sy >= y2:
                limit = sy
    extended = False
    if not trimmed and y2 >= 0.55 * H and y2 < limit - 40:
        y2 = limit
        extended = True
    w, h = x2 - x1, y2 - y1
    bottom_cut = trimmed or extended or (y2 >= limit - 60)
    box = (x1, y1, w, h)
    # YOLO's own instance silhouette travels with the card: SAM2 occasionally
    # locks onto a sliver hypothesis even in multimask mode; compose swaps to
    # this mask whenever the SAM2 result fails coverage sanity.
    yolo_alpha = None
    if mask_best is not None:
        # proto mask spans the full frame at model resolution: crop the box
        # region (scaled) FIRST, then resize — resizing the uncropped proto
        # squeezes the subject into a corner of the card
        ph, pw = mask_best.shape[:2]
        sx, sy = pw / W, ph / H
        c0, c1 = int(x1 * sx), int(np.ceil((x1 + w) * sx))
        r0, r1 = int(y1 * sy), int(np.ceil((y1 + h) * sy))
        try:
            rm = mask_best[max(0, r0):max(1, r1), max(0, c0):max(1, c1)]
            rm = cv2.resize(rm.astype(np.float32), (max(1, w), max(1, h)))
            ya = (rm > 0.5).astype(np.uint8) * 255
            yolo_alpha = np.ascontiguousarray(ya)
        except Exception as e:
            log.warning("yolo alpha build failed: %s", e)
    meta = {"rel_w": w / W, "rel_cx": (x1 + w / 2) / W,
            "rel_cy": (y1 + h / 2) / H,
            "backdrop_white": _backdrop_is_white(img, box),
            "bottom_cut": bottom_cut, "detector": "yolo-structural",
            "conf": round(_conf, 2)}
    if yolo_alpha is not None:
        meta["yolo_alpha"] = yolo_alpha
    cards = [LayerCard(
        "cutout", box, img[y1:y2, x1:x2], _feather_alpha(w, h, 3), z=10,
        meta=meta)]
    log.info("anchor picked structurally: %s conf=%.2f white=%s cut=%s",
             box, _conf, meta["backdrop_white"], bottom_cut)
    return cards


def _resolve_overlaps(cards):
    """Priority banner > strip > corner_logo > cutout; drop smaller overlaps.

    One sacrifice rule: a clashing corner_logo yields to an already-kept
    cutout — logo blobs are the least reliable detection and must never
    evict the program's foreground subject.
    """
    prio = {"banner": 0, "strip": 1, "corner_logo": 2, "cutout": 3}
    keep = []
    for c in sorted(cards, key=lambda c: prio[c.kind]):
        bx, by, bw, bh = c.box
        clash = False
        yield_to = None
        for k in keep:
            kx, ky, kw, kh = k.box
            ix = max(0, min(bx + bw, kx + kw) - max(bx, kx))
            iy = max(0, min(by + bh, ky + kh) - max(by, ky))
            if ix * iy > 0.3 * min(bw * bh, kw * kh):
                # least-reliable detection: a logo blob never wins — neither
                # against a late-arriving anchor nor anyone else
                if c.kind == "corner_logo":
                    clash = True
                    break
                if k.kind == "corner_logo" and c.kind == "cutout":
                    yield_to = k      # evict the logo, admit the subject
                    continue
                clash = True
                break
        if not clash:
            if yield_to is not None:
                keep.remove(yield_to)
            keep.append(c)
    return keep


def _detect_anchor(img, hsv, y_limit=None):
    """Anchor over a flat white backdrop at a frame edge (no YOLO needed).

    News frames paste the presenter over a uniform white column at the
    left or right edge; the anchor is the non-white content inside it.
    YOLO misses white-background cutout presenters surprisingly often.
    y_limit: measure only rows above the strip so the strip's own white
    plate doesn't inflate the column whiteness profile.
    """
    H, W = img.shape[:2]
    ylim = y_limit or H
    white = ((hsv[:ylim, :, 2] > 200) &
             (hsv[:ylim, :, 1] < 60)).astype(np.uint8)
    colfrac = white.mean(axis=0)
    best = None
    for edge in ("left", "right"):
        cols = colfrac if edge == "left" else colfrac[::-1]
        run = 0
        for f in cols:
            if f > 0.25:
                run += 1
            else:
                break
        if run < 0.10 * W:
            continue
        x0 = 0 if edge == "left" else W - run
        content = (white[:, x0:x0 + run] < 128).astype(np.uint8)
        content = cv2.morphologyEx(content, cv2.MORPH_OPEN,
                                   np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(content)
        if n < 2:
            continue
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        bx, by, bw, bh = (int(stats[j, k]) for k in range(4))
        bx += x0
        if bw < 0.08 * W or bh < 0.35 * H:
            continue
        if best is None or bw * bh > best[2] * best[3]:
            best = (bx, by, bw, bh)
    return best


def detect_layers(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    banners = _detect_banners(img, hsv)
    strips = _detect_strips(img, hsv, gray)
    logos = _detect_corner_logo(img, hsv)
    photo_rect = _detect_photo_rect(img)
    strip_box = strips[0].box if strips else None
    cutouts = _detect_cutouts(img, gray, photo_rect, strip_box)
    cards = banners + strips + logos + cutouts
    if not any(c.kind == "cutout" for c in cards):
        # last-resort fallback: uniform-white edge column (before.jpeg-era
        # signal). Kept because it still rescues plain white frames cheaply.
        anchor_box = _detect_anchor(img, hsv,
                                    strip_box[1] if strip_box else None)
        if anchor_box is not None:
            x, y, w, h = anchor_box
            box = (x, y, w, h)
            cards.append(LayerCard(
                "cutout", box, img[y:y + h, x:x + w], _feather_alpha(w, h, 3),
                z=10, meta={"rel_w": w / W, "rel_cx": (x + w / 2) / W,
                            "rel_cy": (y + h / 2) / H, "backdrop_white": True,
                            "bottom_cut": True,
                            "detector": "white-backdrop"}))
            log.info("anchor via white-backdrop fallback: %s", (box,))
    cards = _resolve_overlaps(cards)
    union = np.zeros((H, W), np.uint8)
    subject_union = np.zeros((H, W), np.uint8) if any(
        c.kind == "cutout" for c in cards) else None
    for c in cards:
        x, y, w, h = c.box
        if c.kind == "cutout":
            # erase the anchor zone from the photo band so he never ghosts
            # into the zoom crop on textured backdrops
            pad = 6
            y0, y1_ = max(0, y - pad), min(H, y + h + pad)
            x0, x1_ = max(0, x - pad), min(W, x + w + pad)
            subject_union[y0:y1_, x0:x1_] = 255
        else:
            union[y:y + h, x:x + w] = 255
    log.info("layers: %s photo_rect=%s",
             [(c.kind, c.box) for c in cards], photo_rect)
    return LayerPlan(cards=cards, graphic_union=union, found=bool(cards),
                     photo_rect=photo_rect, subject_union=subject_union)


def clean_photo(img, plan):
    """Inpaint graphic + anchor holes out of the photo band.

    LaMa first: large removed regions need structure-aware filling at full
    resolution — MI-GAN's fixed 512px pass leaves ghosts.
    """
    if not plan.found:
        return img
    union = plan.graphic_union
    if getattr(plan, "subject_union", None) is not None:
        union = np.maximum(union, plan.subject_union)
    from pipeline.fillers import lama_fill
    try:
        filled = lama_fill.fill(img, union)
        engine = "big-lama"
    except Exception as e:
        log.warning("lama cleanup failed (%s) -> migan", e)
        from pipeline.fillers import migan_fill
        filled = migan_fill.fill(img, union)
        engine = "migan"
    keep = union < 128
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
    cutout = next((c for c in plan.cards if c.kind == "cutout"), None)

    # Plate stack metrics first: placements are interlocked (the anchor
    # grounds ON the strip line), so decide geometry before emitting cards.
    gap = 26
    bw_ = bh_ = sw_ = sh_ = 0
    if banner is not None:
        bw_ = int(canvas_w * 0.92)
        bh_ = int(banner.crop.shape[0] * bw_ / banner.crop.shape[1])
    if strip is not None:
        sw_ = int(canvas_w * 0.70)
        sh_ = int(strip.crop.shape[0] * sw_ / strip.crop.shape[1])
    band_bottom = int(config.SMART_PHOTO_HEIGHT * canvas_h)
    avail = canvas_h - band_bottom - 56
    tight = bool((bh_ or sh_) and bh_ + sh_ + 2 * gap <= avail)
    y_banner = y_strip = None
    if tight and (banner or strip):
        top = band_bottom + 24          # hug the photo band like afters do
        if strip:
            y_strip = top
            top += sh_ + gap
        if banner:
            y_banner = top
    else:                               # legacy bottom-anchored stack
        if banner:
            y_banner = canvas_h - bh_ - 36
        if strip:
            y_strip = ((y_banner - sh_ - gap - 20) if y_banner is not None
                       else canvas_h - sh_ - 36)

    if cutout is not None:
        # centered foreground hero, sized by explicit vertical targets:
        # top lands at ~half the photo band, bottom tucks just behind the
        # strip top — reads as double-tall versus the old side placement.
        strip_top = y_strip if y_strip is not None else band_bottom
        visible = strip_top - int(0.46 * band_bottom)
        h = int(visible / 0.92)
        h = min(h, int(0.80 * canvas_h))
        w = int(cutout.crop.shape[1] * h / cutout.crop.shape[0])
        if w > int(0.66 * canvas_w):
            w = int(0.66 * canvas_w)
            h = int(cutout.crop.shape[0] * w / cutout.crop.shape[1])
        x = (canvas_w - w) // 2
        y = max(8, strip_top + int(0.08 * h) - h)
        placed.append((cutout, x, y, w, h))

    if banner is not None:
        placed.append((banner, (canvas_w - bw_) // 2, y_banner, bw_, bh_))
    if strip is not None:
        placed.append((strip, (canvas_w - sw_) // 2, y_strip, sw_, sh_))
    if logo is not None:
        w = int(min(0.20 * canvas_w,
                    max(0.08 * canvas_w, logo.meta["rel_w"] * canvas_w)))
        h = int(logo.crop.shape[0] * w / logo.crop.shape[1])
        placed.append((logo, canvas_w - w - 24, 24, w, h))
    return placed