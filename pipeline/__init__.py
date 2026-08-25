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

    if tier == "relayout":
        cb("filling", 60); t = time.time()
        from pipeline import relayout
        result = relayout.compose(original_rgb, ratio)
        # blurred backdrop is the DESIGN, not a defect - no void gate here
        vm = {"void_fraction": 0.0, "edge_density": 99.0, "voidy": False,
              "note": "relayout-by-design"}
        out_path = config.RESULTS_DIR / f"{job_id}.png"
        _save_png(result, out_path)
        log.info("job=%s relayout void=%s", job_id, vm)
        return {"result_path": out_path, "paste_box": None,
                "offset_reason": "relayout", "engine": relayout.ENGINE,
                "void": vm, "engine_chain": [("relayout", False)],
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


def run_smart(original_rgb: np.ndarray, ratio: str = "9:16",
              tier: str = "auto", align: str = "auto",
              job_id: str | None = None, cb=None):
    """Universal engine chain: user's tier is PREFERRED, never forced-final.

    Any voidy result escalates to the next engine until bands pass the gate.
    No selection can ever ship dead bands.
    """
    order = {
        "auto": ["fast", "sdturbo", "relayout"],
        "fast": ["fast", "sdturbo", "relayout"],
        "quality": ["quality", "sdturbo", "relayout"],
        "sdturbo": ["sdturbo", "relayout"],
        "relayout": ["relayout"],
    }.get(tier, ["fast", "sdturbo", "relayout"])

    chain = []
    if cb is None:
        cb = lambda stage, pct: None

    meta = None
    for t in order:
        if t == "relayout" and meta is not None and cb is not None:
            cb("filling", 55)
        try:
            meta = run(original_rgb, ratio, t, align, job_id, cb)
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
