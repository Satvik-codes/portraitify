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

    cb("filling", 60); t = time.time()
    canvas = prefill_mirror(canvas, photo_box)
    filled, engine = None, None
    for fill_tier in config.SMART_FILL_ORDER:
        try:
            if fill_tier == "sdturbo":
                from pipeline.fillers import sdturbo_fill
                filled = sdturbo_fill.fill(canvas, mask)
                engine = sdturbo_fill.ENGINE
            else:
                from pipeline.fillers import lama_fill
                filled = lama_fill.fill(canvas, mask)
                engine = lama_fill.ENGINE
            break
        except Exception as e:
            log.warning("smart fill %s failed: %s", fill_tier, e)
    if filled is None:
        raise RuntimeError("all smart fill engines failed")
    from pipeline import prefill
    filled = prefill.scrub_placeholder(filled, photo_box)
    timings["fill"] = round(time.time() - t, 3)

    cb("composing", 92); t = time.time()
    import cv2 as _cv2
    from pipeline.compose import composite, assert_region_identical
    from pipeline import cards as C
    from pipeline import cutout as CUT
    result = composite(filled, band, photo_box)
    assert_region_identical(result, band, photo_box)

    footprints = []
    fidelity_ok = True
    for card, x, y, w, h in placed:
        target = cv2_resize(card.crop, w, h)
        if card.kind == "banner":
            alpha = np.full((h, w), 255, np.uint8)
        else:
            try:
                alpha = CUT.box_alpha(original_rgb, card.box)
                alpha = _cv2.resize(alpha, (w, h), interpolation=_cv2.INTER_LINEAR)
            except Exception as e:
                log.warning("SAM2 alpha failed for %s (%s) -> solid", card.kind, e)
                alpha = np.full((h, w), 255, np.uint8)
        fp = C.paste_exact(result, target, alpha, x, y)
        # fidelity: solid pixels must equal the deterministic scaled crop
        solid = alpha >= 250
        if not np.array_equal(result[fp[1]:fp[1] + fp[3], fp[0]:fp[0] + fp[2]][solid],
                              target[solid]):
            fidelity_ok = False
        footprints.append(fp)
    assert fidelity_ok, "graphic card fidelity violated"

    from pipeline import voidcheck
    vm = voidcheck.band_metrics(result, photo_box, exclude_boxes=footprints)
    timings["compose"] = round(time.time() - t, 3)
    timings["total"] = round(time.time() - t_all, 3)

    out_path = config.RESULTS_DIR / f"{job_id}.png"
    _save_png(result, out_path)
    log.info("job=%s smart cards=%d engine=%s void=%s timings=%s",
             job_id, len(placed), engine, vm, timings)
    return {"result_path": out_path, "paste_box": photo_box,
            "smart_crop_x": x_crop if sw >= cw else 0,
            "offset_reason": f"smart|{method}", "engine": f"smart({engine})",
            "void": vm, "cards_composed": len(placed),
            "fidelity_ok": fidelity_ok,
            "engine_chain": [("smart/" + engine, vm["voidy"])],
            "timings": timings}


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
    meta["engine_chain"] = chain
    return meta


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
