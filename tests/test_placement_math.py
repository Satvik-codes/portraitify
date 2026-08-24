"""Phase-2 geometry tests. Run: python tests\\test_placement_math.py (or pytest)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from config import config
from pipeline.placement import decide_paste, build_mask
from pipeline.compose import composite, assert_region_identical

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"{name} FAILED {detail}"
    PASS += 1
    print(f"ok   {name}")


def test_landscape_fit_width_center():
    d = decide_paste(1920, 1080, 1080, 1920, "center")
    x, y, w, h = d.paste_box
    check("landscape w==canvas", w == 1080)
    check("landscape h scaled", abs(h - 608) <= 1, f"h={h}")
    check("landscape centered", y == (1920 - h) // 2)


def test_auto_thirds_bias():
    d = decide_paste(1920, 1080, 1080, 1920, "auto", subject_centroid_y_norm=0.9)
    x, y, w, h = d.paste_box
    target = config.AUTO_CENTROID_TARGET * 1920
    got = y + 0.9 * h
    check("auto maps centroid to ~42%", abs(got - target) <= max(3.0, 0.9), f"got={got:.1f} target={target:.1f}")
    check("auto clamped inside", 0 <= y <= 1920 - h)


def test_panorama_fit_width_big_bands():
    d = decide_paste(3000, 400, 1080, 1920, "center")
    _, _, w, h = d.paste_box
    check("panorama w==canvas", w == 1080 and h == round(400 * 1080 / 3000))


def test_tall_portrait_fit_height():
    d = decide_paste(1000, 3000, 1080, 1080, "center")
    x, y, w, h = d.paste_box
    check("tall fit-height", h == 1080 and w == 360, f"{w}x{h}")
    check("tall centered x", x == 360 and y == 0)


def test_align_top_bottom():
    top = decide_paste(1920, 1080, 1080, 1920, "top").paste_box
    bot = decide_paste(1920, 1080, 1080, 1920, "bottom").paste_box
    check("align top", top[1] == 0)
    check("align bottom", bot[1] + bot[3] == 1920)


def test_mask_invariant():
    cw, ch = config.CANVAS_SIZES["9:16"]
    box = (0, 656, 1080, 608)  # touches left+right borders on purpose
    m = build_mask(cw, ch, box)
    x, y, w, h = box
    interior = m[y + 8:y + h - 8, x + 8:x + w - 8]
    check("mask protects interior", int(interior.max()) == 0)
    open_area = m[:200, :]
    check("mask fills open area", int(open_area.min()) >= 128)
    check("mask has soft seam", int(m.std()) > 10, f"std={m.std():.1f}")


def test_composite_guarantee_and_ramp():
    rng = np.random.default_rng(7)
    H, W = 400, 300
    box = (50, 60, 120, 90)
    filled = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    orig = rng.integers(0, 255, (box[3], box[2], 3), dtype=np.uint8)
    out0 = composite(filled, orig, box, ramp_px=0)
    assert_region_identical(out0, orig, box)
    check("composite exact no-ramp", True)
    out16 = composite(filled, orig, box, ramp_px=12)
    assert_region_identical(out16, orig, box)
    x, y, w, h = box
    ring_changed = not np.array_equal(out16[y+h:y+h+6, x:x+w], filled[y+h:y+h+6, x:x+w])
    check("ramp modifies outer ring", ring_changed)
    far_same = np.array_equal(out16[:40, :], filled[:40, :])
    check("ramp leaves far field", far_same)


if __name__ == "__main__":
    for fn in [test_landscape_fit_width_center, test_auto_thirds_bias,
               test_panorama_fit_width_big_bands, test_tall_portrait_fit_height,
               test_align_top_bottom, test_mask_invariant, test_composite_guarantee_and_ramp]:
        fn()
    print(f"\nALL {PASS} CHECKS PASSED")
