"""Canvas placement + mask geometry. Pure functions, no ML imports."""
from dataclasses import dataclass
import numpy as np
import cv2

from config import config


@dataclass
class PlacementDecision:
    paste_box: tuple  # (x, y, w, h) ints on canvas
    subject_centroid_y: float | None
    offset_reason: str


def decide_paste(orig_w: int, orig_h: int, canvas_w: int, canvas_h: int,
                 align: str = "auto", subject_centroid_y_norm: float | None = None) -> PlacementDecision:
    """Scale original uniformly to fit the canvas and choose its position.

    Fit-width first; if that overflows canvas height (portrait-ish input into
    short canvas), fall back to fit-height centered horizontally.
    """
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"bad source dims {orig_w}x{orig_h}")

    # Contain-scale: works for landscape, portrait, and already-vertical inputs
    scale = min(canvas_w / orig_w, canvas_h / orig_h)

    w = max(1, int(round(orig_w * scale)))
    h = max(1, int(round(orig_h * scale)))
    x = (canvas_w - w) // 2  # safe in both modes

    if align == "top":
        y, reason = 0, "user_align_top"
    elif align == "bottom":
        y, reason = canvas_h - h, "user_align_bottom"
    elif align == "center" or subject_centroid_y_norm is None:
        y = (canvas_h - h) // 2
        reason = ("auto_no_subject" if align == "auto" else "centered")
    else:
        c = float(np.clip(subject_centroid_y_norm, 0.0, 1.0))
        target_px = config.AUTO_CENTROID_TARGET * canvas_h
        y = int(round(target_px - c * h))
        y = int(np.clip(y, 0, canvas_h - h))
        reason = f"auto_thirds(c={c:.2f})"

    return PlacementDecision((x, y, w, h), subject_centroid_y_norm, reason)


def build_mask(canvas_w: int, canvas_h: int, paste_box: tuple) -> np.ndarray:
    """uint8 (H,W): 255 = fill here, 0 = protect.

    Open canvas is 255; an OVERLAP_PX-thick band just inside the paste border
    is forced back to 255 so fillers see real context across the seam; after
    Gaussian feathering, the interior (minus a 2 px rim) is re-solidified to
    hard 0 so original pixels are never touched.
    """
    H, W = canvas_h, canvas_w
    x, y, w, h = paste_box
    x, y = max(0, x), max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)
    if w <= 0 or h <= 0:
        raise ValueError(f"paste_box out of canvas: {(x, y, w, h)}")

    mask = np.ones((H, W), dtype=np.float32)

    rim = 2
    ix0, iy0 = x + rim, y + rim
    ix1, iy1 = x + w - rim, y + h - rim
    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = 0.0

    t = max(1, min(config.OVERLAP_PX, min(w, h) // 3))
    mask[y:y + t, x:x + w] = 1.0            # top band
    mask[y + h - t:y + h, x:x + w] = 1.0    # bottom band
    mask[y:y + h, x:x + t] = 1.0            # left band
    mask[y:y + h, x + w - t:x + w] = 1.0    # right band

    k = config.FEATHER_PX | 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    if ix1 > ix0 and iy1 > iy0:
        mask[iy0:iy1, ix0:ix1] = 0.0  # hard protection survives the blur

    return (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)


if __name__ == "__main__":
    import sys
    from PIL import Image

    img_path = sys.argv[1]
    ratio = sys.argv[2] if len(sys.argv) > 2 else "9:16"
    img = Image.open(img_path).convert("RGB")
    cw, ch = config.CANVAS_SIZES[ratio]
    dec = decide_paste(img.width, img.height, cw, ch, "auto", 0.5)
    m = build_mask(cw, ch, dec.paste_box)
    canvas = np.full((ch, cw, 3), 127, np.uint8)
    px, py, pw, ph = dec.paste_box
    canvas[py:py + ph, px:px + pw] = np.array(img.resize((pw, ph)))
    overlay = canvas.copy()
    sel = m > 128
    overlay[sel] = (overlay[sel].astype(np.float32) * 0.45 +
                    np.float32([0, 120, 255]) * 0.55).astype(np.uint8)
    out = config.DEBUG_DIR / "placement_demo.jpg"
    Image.fromarray(overlay).save(out, quality=90)
    print(dec)
    print("demo ->", out)
