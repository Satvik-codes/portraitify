"""Pipeline orchestrator: detect -> place -> fill -> compose -> save.

Stages report progress through cb(stage:str, pct:int).
The pixel guarantee is asserted on every run before the result is saved.
"""
import logging
import time
import uuid

import numpy as np

from config import config

log = logging.getLogger("h2v.pipeline")


class QualityNotInstalledError(RuntimeError):
    pass


class NoGraphicsSignal(RuntimeError):
    """Raised by smart tier when the image has no graphic layers."""


def run(original_rgb: np.ndarray, ratio: str = "9:16", tier: str = "fast",
        align: str = "auto", job_id: str | None = None, cb=None):
    """Returns dict(result_path, paste_box, offset_reason, engine, timings)."""
    t_all = time.time()
    job_id = job_id or uuid.uuid4().hex
    timings = {}
    if cb is None:
        cb = lambda stage, pct: None

    H, W = original_rgb.shape[:2]
    cw, chh = config.CANVAS_SIZES[ratio]

    if tier == "smart":
        return _run_smart_layered(original_rgb, ratio, align, job_id, cb, cw, chh)

    if tier == "relayout":
        cb("filling", 60); t = time.time()
        from pipeline import relayout
        result, info = relayout.compose(original_rgb, ratio)
        # blurred backdrop is the DESIGN, not a defect - no void gate here
        vm = {"void_fraction": 0.0, "edge_density": 99.0, "voidy": False,
              "note": "relayout-by-design"}
        out_path = config.RESULTS_DIR / f"{job_id}.png"
        _save_png(result, out_path)
        log.info("job=%s relayout info=%s", job_id, info)
        return {"result_path": out_path, "paste_box": None,
                "offset_reason": "relayout", "engine": relayout.ENGINE,
                "void": vm, "cards_composed": info["cards_composed"],
                "fidelity_ok": info["fidelity_ok"],
                "engine_chain": [("relayout", False)],
                "timings": {"total": round(time.time() - t_all, 3)}}

    # ---- detect -------------------------------------------------------------
    cb("detecting", 10); t = time.time()
    from pipeline.detect import subject_centroid_y
    centroid, method = subject_centroid_y(original_rgb)
    timings["detect"] = round(time.time() - t, 3)

    # ---- place --------------------------------------------------------------
    cb("placing", 25); t = time.time()
    from pipeline.placement import decide_paste, build_mask
    dec = decide_paste(W, H, cw, chh, align, centroid)
    px, py, pw, ph = dec.paste_box
    scaled = _resize_exact(original_rgb, pw, ph)
    canvas = np.full((chh, cw, 3), 127, dtype=np.uint8)
    canvas[py:py + ph, px:px + pw] = scaled
    mask = build_mask(cw, chh, dec.paste_box)
    timings["place"] = round(time.time() - t, 3)

    # ---- fill ---------------------------------------------------------------
    cb("filling", 70); t = time.time()
    from pipeline import prefill
    if tier == "fast":
        canvas = prefill.mirror_pad(canvas, dec.paste_box)  # structured seed
        from pipeline.fillers import lama_fill
        filled = lama_fill.fill(canvas, mask)
        engine = lama_fill.ENGINE
    elif tier == "quality":
        from pipeline.fillers import powerpaint_fill
        if not powerpaint_fill.available():
            raise QualityNotInstalledError(
                "Quality tier needs PowerPaint-V2 weights. "
                "Run scripts\\download_powerpaint.ps1 first.")
        filled = powerpaint_fill.fill(canvas, mask, scaled, dec.paste_box)
        engine = powerpaint_fill.ENGINE
    elif tier == "sdturbo":
        from pipeline.fillers import sdturbo_fill
        if not sdturbo_fill.available():
            raise QualityNotInstalledError(
                "SD-Turbo engine needs the cached stabilityai/sd-turbo "
                "snapshot (present in %USERPROFILE%\\.cache\\huggingface).")
        canvas = prefill.mirror_pad(canvas, dec.paste_box)
        filled = sdturbo_fill.fill(canvas, mask)
        engine = sdturbo_fill.ENGINE
    else:
        raise ValueError(f"unknown tier '{tier}'")
    filled = prefill.scrub_placeholder(filled, dec.paste_box)
    timings["fill"] = round(time.time() - t, 3)

    # ---- compose (guarantee enforced) ----------------------------------------
    cb("composing", 98); t = time.time()
    from pipeline.compose import composite, assert_region_identical
    result = composite(filled, scaled, dec.paste_box)
    assert_region_identical(result, scaled, dec.paste_box)  # hard gate
    from pipeline import voidcheck
    meta_void = voidcheck.band_metrics(result, dec.paste_box)
    timings["compose"] = round(time.time() - t, 3)
    timings["total"] = round(time.time() - t_all, 3)

    out_path = config.RESULTS_DIR / f"{job_id}.png"
    _save_png(result, out_path)

    _dump_debug(job_id, original_rgb, canvas, mask, filled, result, dec) if config.KEEP_DEBUG else None

    log.info("job=%s %s %s engine=%s method=%s box=%s void=%s timings=%s",
             job_id, ratio, tier, engine, method, dec.paste_box, meta_void, timings)
    return {"result_path": out_path, "paste_box": dec.paste_box,
            "offset_reason": f"{dec.offset_reason}|{method}",
            "engine": engine, "void": meta_void, "timings": timings}


def _run_smart_layered(original_rgb, ratio, align, job_id, cb, cw, chh):
    """Layer-aware recomposition: photo dominates + graphics re-composed."""
    t_all = time.time()
    timings = {}
    t = time.time()
    from pipeline import layers as L
    plan = L.detect_layers(original_rgb)
    timings["detect"] = round(time.time() - t, 3)
    if not plan.found:
        raise NoGraphicsSignal("no graphic layers detected")

    cb("cleaning", 15); t = time.time()
    cleaned = L.clean_photo(original_rgb, plan)
    # Crop to the dominant photo rect when found, then drop the recomposed
    # anchor's home columns entirely: their inpainted patch smears on
    # textured backdrops and carries no story value. Mid-frame anchors are
    # spared (their backdrop IS the scene); a minimum band stays guaranteed.
    _co = next((c for c in plan.cards if c.kind == "cutout"), None)
    if getattr(plan, "photo_rect", None):
        rx, ry, rw, rh = plan.photo_rect
        cleaned = cleaned[ry:ry + rh, rx:rx + rw]
    if _co is not None:
        _x0, _y0, _w0, _h0 = _co.box
        Ws = cleaned.shape[1]
        pad = int(0.02 * Ws)
        if (_x0 + _w0 / 2) < 0.55 * Ws and _x0 < 0.30 * Ws:
            sx = _x0 + _w0 + pad
            if Ws - sx >= 320:
                cleaned = cleaned[:, sx:]
        elif (_x0 + _w0 / 2) > 0.45 * Ws and (_x0 + _w0) > 0.70 * Ws:
            ex = max(320, _x0 - pad)
            if ex >= 320:
                cleaned = cleaned[:, :ex]
    timings["clean"] = round(time.time() - t, 3)

    cb("placing", 25); t = time.time()
    from pipeline.detect import subject_centroid_y
    centroid, method = subject_centroid_y(cleaned)
    from pipeline.placement import build_mask

    # after.png reference: photo zoom-cropped to fill ~72% height,
    # top-anchored full-width; graphics stack in the bottom band.
    target_h = max(64, int(config.SMART_PHOTO_HEIGHT * chh))
    scale = target_h / cleaned.shape[0]
    sw = int(round(cleaned.shape[1] * scale))
    if sw >= cw:
        scaled = _resize_exact(cleaned, sw, target_h)
        x_crop = (sw - cw) // 2
        band = scaled[:, x_crop:x_crop + cw]
        photo_box = (0, 0, cw, target_h)
    else:  # image too tall relative to canvas -> fit width, top anchored
        fh = min(chh - 64, int(round(cleaned.shape[0] * cw / cleaned.shape[1])))
        scaled = _resize_exact(cleaned, cw, fh)
        band = scaled
        photo_box = (0, 0, cw, fh)
        target_h = fh
    canvas = np.full((chh, cw, 3), 127, dtype=np.uint8)
    canvas[0:target_h] = band
    mask = build_mask(cw, chh, photo_box)
    placed = L.plan_composition(plan, cw, chh, centroid)
    timings["place"] = round(time.time() - t, 3)

    cb("backdrop", 60); t = time.time()
    # Designed backdrop: vertical gradient from the photo's bottom-edge
    # colors, darkened toward the bottom. Deterministic, instant, and it
    # reads as an intentional poster background — no generation, no voids.
    band_bottom = band[-12:].mean(axis=(0, 1)).astype(np.float32)
    band_h = chh - target_h
    tt = np.linspace(0.0, 1.0, max(1, band_h), dtype=np.float32)[:, None, None]
    grad = (band_bottom[None, None] * (1 - 0.62 * tt)).astype(np.uint8)
    canvas[target_h:] = grad
    timings["backdrop"] = round(time.time() - t, 3)

    cb("composing", 92); t = time.time()
    import cv2 as _cv2
    from pipeline import cards as C
    from pipeline import cutout as CUT
    result = canvas

    footprints = []
    fidelity_ok = True
    for card, x, y, w, h in placed:
        target = cv2_resize(card.crop, w, h)
        if card.kind == "banner" or card.kind == "strip":
            # solid plate: the strip's own white design reads cleanly on the
            # dark backdrop; keying metallic letters to transparency turns
            # them black (dark halos dominate on dark backgrounds)
            alpha = np.full((h, w), 255, np.uint8)
        else:
            try:
                alpha = _pick_cutout_alpha(CUT, original_rgb, card, w, h)
                # halo cleanup ONLY on the mask's outer rim band: light-gray
                # JPEG mush around white-backdrop anchors would otherwise
                # glare on the red gradient. Interior pixels — shirts,
                # skin, everything — keep their exact source colors.
                if card.meta.get("backdrop_white"):
                    # the source strokes white-backdrop anchors with a dark
                    # keyline — erode past it; drop floating backdrop
                    # islands (stray patches between arm gaps)
                    alpha = _cv2.erode(alpha, np.ones((9, 9), np.uint8))
                    am = alpha > 128
                    n_, lab_, st_, _ = _cv2.connectedComponentsWithStats(
                        am.astype(np.uint8), 8)
                    if n_ > 1:
                        j_ = 1 + int(np.argmax(st_[1:, _cv2.CC_STAT_AREA]))
                        main_a = float(st_[j_, _cv2.CC_STAT_AREA])
                        for i_ in range(1, n_):
                            if (i_ != j_ and
                                    st_[i_, _cv2.CC_STAT_AREA] < 0.04 * main_a):
                                alpha[lab_ == i_] = 0
                    hsvt = _cv2.cvtColor(target, _cv2.COLOR_RGB2HSV)
                    white = (hsvt[..., 2] > 165) & (hsvt[..., 1] < 85)
                    core = _cv2.erode(alpha, np.ones((7, 7), np.uint8))
                    rim = (alpha > 0) & (core == 0)
                    alpha = np.where(rim & white, 0, alpha)
                    alpha = _cv2.GaussianBlur(alpha, (3, 3), 0)
                # soften ground-out bodies (frame edge / strip line)
                if card.meta.get("bottom_cut", True):
                    fh2 = max(1, int(0.12 * h))
                    fade = np.ones(h, np.float32)
                    fade[-fh2:] = np.linspace(1.0, 0.0, fh2, dtype=np.float32)
                    alpha = (alpha.astype(np.float32) * fade[:, None]).astype(np.uint8)
            except Exception as e:
                log.warning("cutout alpha failed for %s (%s) -> solid", card.kind, e)
                alpha = np.full((h, w), 255, np.uint8)
        fp = C.paste_exact(result, target, alpha, x, y)
        # fidelity: solid pixels must equal the deterministic scaled crop
        solid = alpha >= 250
        if not np.array_equal(result[fp[1]:fp[1] + fp[3], fp[0]:fp[0] + fp[2]][solid],
                              target[solid]):
            fidelity_ok = False
        footprints.append(fp)
    assert fidelity_ok, "graphic card fidelity violated"

    # Backdrop is designed by construction — the void gate does not apply.
    vm = {"void_fraction": 0.0, "edge_density": -1, "voidy": False,
          "note": "designed-backdrop"}
    timings["compose"] = round(time.time() - t, 3)
    timings["total"] = round(time.time() - t_all, 3)

    out_path = config.RESULTS_DIR / f"{job_id}.png"
    _save_png(result, out_path)
    log.info("job=%s smart cards=%d fidelity=%s timings=%s",
             job_id, len(placed), fidelity_ok, timings)
    return {"result_path": out_path, "paste_box": photo_box,
            "smart_crop_x": x_crop if sw >= cw else 0,
            "offset_reason": f"smart|{method}", "engine": "smart(layers)",
            "void": vm, "cards_composed": len(placed),
            "fidelity_ok": fidelity_ok,
            "engine_chain": [("smart/layers", False)],
            "timings": timings}


def _alpha_ok(a, w, h, cov_lo=0.10):
    """A plausible person mask: non-trivial coverage AND spans the box.

    Sliver masks (a face with no torso) pass coverage alone — span gates
    catch them; near-solid rectangles (busy fill fallback) fail cov_hi.
    """
    am = a > 128
    ys, xs = np.where(am)
    if not xs.size:
        return False
    cov = float(am.mean())
    bw_a = int(xs.max() - xs.min() + 1)
    bh_a = int(ys.max() - ys.min() + 1)
    return (cov_lo <= cov <= 0.80 and
            bw_a >= 0.45 * w and bh_a >= 0.50 * h)


def _pick_cutout_alpha(CUT, img_rgb, card, w, h):
    """Ranked alpha ladder; first sane candidate wins, models degrade into
    geometry before anything ever ships broken."""
    import cv2
    tried = []
    try:
        sam, union = CUT.box_alpha_multi(img_rgb, card.box)
        for tag, a in (("sam", sam), ("union", union)):
            ar = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            if _alpha_ok(ar, w, h):
                log.info("cutout alpha via %s", tag)
                return ar
            tried.append(tag)
    except Exception as e:
        log.warning("SAM2 ladder failed (%s)", e)
    ym = card.meta.get("yolo_alpha")
    if ym is not None:
        ar = cv2.resize(np.asarray(ym), (w, h),
                        interpolation=cv2.INTER_LINEAR)
        if _alpha_ok(ar, w, h):
            log.info("cutout alpha via yolo silhouette")
            return ar
        tried.append("yolo")
    try:
        ar = CUT.box_alpha_fill(img_rgb, card.box)
        ar = cv2.resize(ar, (w, h), interpolation=cv2.INTER_LINEAR)
        if _alpha_ok(ar, w, h, cov_lo=0.05):
            log.warning("cutout alpha via geometric fill (all model masks "
                        "degenerate: %s)", ",".join(tried) or "none")
            return ar
        log.warning("fill fallback failed sanity (cov=%.2f)",
                    float((ar > 128).mean()))
    except Exception as e:
        log.warning("fill fallback failed (%s)", e)
    try:
        # near-solid fills mean busy art beat the color model — GrabCut's
        # layout priors are the last content-aware rescue before the floor
        ar = CUT.box_alpha_grabcut(img_rgb, card.box)
        ar = cv2.resize(ar, (w, h), interpolation=cv2.INTER_LINEAR)
        if _alpha_ok(ar, w, h, cov_lo=0.05):
            log.warning("cutout alpha via grabcut")
            return ar
    except Exception as e:
        log.warning("grabcut fallback failed (%s)", e)
    # absolute floor: feathered solid of the detector box — an honest card
    # beats a hollow one
    a = np.full((h, w), 255, np.uint8)
    k = 9 | 1
    a[:3] = a[-3:] = a[:, :3] = a[:, -3:] = 0
    return cv2.GaussianBlur(a, (k, k), 0)


def prefill_mirror(canvas, paste_box):
    from pipeline import prefill
    return prefill.mirror_pad(canvas, paste_box)


def cv2_resize(img, w, h):
    import cv2
    interp = cv2.INTER_AREA if (w * h) < img.shape[0] * img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def run_smart(original_rgb: np.ndarray, ratio: str = "9:16",
              tier: str = "auto", align: str = "auto",
              job_id: str | None = None, cb=None):
    """Universal engine chain: user's tier is PREFERRED, never forced-final.

    Any voidy result escalates to the next engine until bands pass the gate.
    No selection can ever ship dead bands.
    """
    order = {
        "auto": ["smart", "fast", "sdturbo", "relayout"],
        "smart": ["smart", "sdturbo", "relayout"],
        "fast": ["fast", "sdturbo", "relayout"],
        "quality": ["quality", "sdturbo", "relayout"],
        "sdturbo": ["sdturbo", "relayout"],
        "relayout": ["relayout"],
    }.get(tier, ["smart", "fast", "sdturbo", "relayout"])

    chain = []
    if cb is None:
        cb = lambda stage, pct: None

    meta = None
    for t in order:
        if t == "relayout" and meta is not None and cb is not None:
            cb("filling", 55)
        try:
            meta = run(original_rgb, ratio, t, align, job_id, cb)
        except NoGraphicsSignal:
            chain.append(("smart", "no-graphics"))
            continue
        except Exception as e:
            log.warning("engine %s failed in chain: %s", t, e)
            chain.append((t, "error"))
            continue
        chain.append((t, meta["void"]["voidy"]))
        if not meta["void"]["voidy"]:
            break
    if meta is None or meta["void"]["voidy"]:
        # no-voids guarantee: the deterministic fit+gradient compose cannot
        # hallucinate or empty out — it is the floor every job lands on.
        log.warning("all engines failed/voidy -> terminal fit compose")
        chain.append(("terminal-fit", False))
        meta = _terminal_fit_compose(original_rgb, ratio, job_id)
    meta["engine_chain"] = chain
    return meta


def _terminal_fit_compose(original_rgb: np.ndarray, ratio: str,
                          job_id: str) -> dict:
    """Deterministic last resort: center-fit band + color-sampled gradient.

    Void-free by construction — no diffusion, no gates that can reject.
    """
    t0 = time.time()
    cw, chh = config.CANVAS_SIZES[ratio]
    target_h = max(64, int(config.SMART_PHOTO_HEIGHT * chh))
    scale = target_h / original_rgb.shape[0]
    sw = int(round(original_rgb.shape[1] * scale))
    if sw >= cw:
        scaled = _resize_exact(original_rgb, sw, target_h)
        x0 = (sw - cw) // 2
        band = scaled[:, x0:x0 + cw]
        paste_box = (0, 0, cw, target_h)
    else:
        fh = min(chh - 64,
                 int(round(original_rgb.shape[0] * cw / original_rgb.shape[1])))
        band = _resize_exact(original_rgb, cw, fh)
        paste_box = (0, 0, cw, fh)
        target_h = fh
    canvas = np.full((chh, cw, 3), 127, dtype=np.uint8)
    canvas[0:target_h] = band
    bb = band[-12:].mean(axis=(0, 1)).astype(np.float32)
    tt = np.linspace(0.0, 1.0, max(1, chh - target_h),
                     dtype=np.float32)[:, None, None]
    canvas[target_h:] = (bb[None, None] * (1 - 0.62 * tt)).astype(np.uint8)
    out_path = config.RESULTS_DIR / f"{job_id}.png"
    _save_png(canvas, out_path)
    vm = {"void_fraction": 0.0, "edge_density": -1, "voidy": False,
          "note": "terminal-fit"}
    return {"result_path": out_path, "paste_box": paste_box,
            "offset_reason": "terminal-fit", "engine": "terminal-fit",
            "void": vm, "cards_composed": 0, "fidelity_ok": True,
            "timings": {"total": round(time.time() - t0, 2)}}


# --- helpers ------------------------------------------------------------------

def _resize_exact(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2
    interp = cv2.INTER_AREA if (w * h) < img.shape[0] * img.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def _save_png(arr: np.ndarray, path):
    import cv2
    ok = cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise IOError(f"failed writing {path}")


def _dump_debug(job_id, original, canvas, mask, filled, result, dec):
    import cv2
    d = config.DEBUG_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "mask.png"), mask)
    for name, arr in [("canvas", canvas), ("filled", filled), ("result", result)]:
        cv2.imwrite(str(d / f"{name}.jpg"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
