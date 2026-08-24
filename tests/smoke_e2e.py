"""End-to-end smoke: converts bundled/synthesized samples, prints timing table.

Usage: python tests\\smoke_e2e.py [--tier fast|quality]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from config import config


def _synth_samples():
    """Self-contained stand-ins when no real photos are dropped in yet."""
    rng = np.random.default_rng(3)
    out = {}

    news = np.zeros((720, 1280, 3), np.uint8)
    news[..., 0] = 40; news[..., 1] = 60; news[..., 2] = 110
    news += rng.integers(-8, 8, news.shape, dtype=np.int16).astype(np.uint8)
    news[560:620, :] = (30, 30, 200)          # red banner
    news[620:680, :] = (240, 240, 240)        # ticker strip
    cv_rect(news, (100, 120, 260, 300))       # anchor box
    out["news_1280x720"] = news

    group = np.full((1080, 1920, 3), 90, np.uint8)
    for cx, cy, r, col in [(480, 700, 150, (150, 170, 220)),
                           (960, 760, 170, (200, 160, 140)),
                           (1440, 690, 155, (130, 190, 150))]:
        blob(group, cx, cy, r, col)
    group += rng.integers(-10, 10, group.shape, dtype=np.int16).astype(np.uint8)
    out["group_1920x1080"] = group

    land = np.zeros((1080, 1920, 3), np.uint8)
    yy, xx = np.mgrid[0:1080, 0:1920]
    land[..., 0] = (yy / 1080 * 120).astype(np.uint8)
    land[..., 1] = (xx / 1920 * 90 + 40).astype(np.uint8)
    land[..., 2] = 180 - (yy / 1080 * 120).astype(np.uint8)
    blob(land, 1450, 300, 110, (250, 230, 160))
    out["landscape_1920x1080"] = land
    return out


def cv_rect(img, rect):
    x, y, w, h = rect
    img[y:y + h, x:x + w] = (60, 130, 60)


def blob(img, cx, cy, r, col):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[m] = col


def load_samples():
    d = Path(__file__).parent / "samples"
    real = {}
    for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png")) + sorted(d.glob("*.jpeg")):
        from PIL import Image
        real[p.stem] = np.array(Image.open(p).convert("RGB"))
    if real:
        return real
    print("[info] tests/samples empty - using synthesized stand-ins")
    return _synth_samples()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="fast", choices=["fast", "quality"])
    args = ap.parse_args()

    import pipeline
    samples = load_samples()
    rows = []
    for name, img in samples.items():
        t0 = time.time()
        meta = pipeline.run(img, ratio="9:16", tier=args.tier,
                            job_id=f"smoke-{name[:24]}")
        res = np.array(Image_open(meta["result_path"]))
        cw, chh = config.CANVAS_SIZES["9:16"]
        ok_size = res.shape[:2] == (chh, cw)
        bands = np.concatenate([res[:meta["paste_box"][1]],
                                res[meta["paste_box"][1] + meta["paste_box"][3]:]])
        # A truly blank row = untouched placeholder gray (mean~127, std~0).
        # Smooth-but-real fills (sky/walls) are flat yet NOT 127-gray.
        rm = bands.reshape(bands.shape[0], -1).astype(np.float32)
        blank_rows = int(((np.abs(rm.mean(axis=1) - 127) < 1.5) &
                          (rm.std(axis=1) < 1.5)).sum())
        rows.append((name, meta["engine"], meta["timings"]["total"],
                     ok_size, blank_rows, meta["offset_reason"]))
        print(f"[{name}] done in {time.time()-t0:.1f}s -> {meta['result_path'].name}")

    print("\n{:28} {:12} {:>7} {:>6} {:>8}  {}".format(
        "sample", "engine", "sec", "size", "flatRows", "placement"))
    for r in rows:
        print("{:28} {:12} {:>7.2f} {:>6} {:>8}  {}".format(*r))

    bad = [r for r in rows if not r[3] or r[4] > 0]
    if bad:
        print(f"\nFAILED checks on: {[b[0] for b in bad]}"); sys.exit(1)
    print("\nSMOKE PASSED")


def Image_open(p):
    from PIL import Image
    return Image.open(p)


if __name__ == "__main__":
    main()
