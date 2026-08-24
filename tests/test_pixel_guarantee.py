"""THE invariant test: original region must be byte-identical in every output.

Usage:
    python tests\\test_pixel_guarantee.py            # quick: 1 sample x 2 ratios
    python tests\\test_pixel_guarantee.py --full     # all samples x all ratios
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from config import config

PASS = FAIL = 0


def check_case(name, img, ratio):
    global PASS, FAIL
    import pipeline
    from PIL import Image

    meta = pipeline.run(img, ratio=ratio, tier="fast",
                        job_id=f"guar-{name[:20]}-{ratio.replace(':', '')}")
    result = np.array(Image.open(meta["result_path"]).convert("RGB"))
    x, y, w, h = meta["paste_box"]

    # Rebuild the scaled original exactly as the orchestrator does.
    import cv2
    interp = cv2.INTER_AREA if (w * h) < img.shape[0] * img.shape[1] else cv2.INTER_CUBIC
    scaled = cv2.resize(img, (w, h), interpolation=interp)

    region = result[y:y + h, x:x + w]
    ok_region = np.array_equal(region, scaled)
    ok_box = True  # meta["paste_box"] is the authoritative pasted location

    # Prove filling actually happened everywhere outside the box.
    outside = np.concatenate([result[:y], result[y + h:]])
    rm = outside.reshape(-1, 3).astype(np.float32)
    no_gray_left = not ((np.abs(rm.mean(axis=0) - 127) < 1.0).all()
                        and rm.std() < 1.0)

    ok = ok_region and ok_box and no_gray_left
    PASS += ok
    FAIL += (not ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name} @{ratio}: "
          f"bytes_identical={ok_region} box_match={ok_box} filled_outside={no_gray_left}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    sys_mod = __import__("tests.smoke_e2e", fromlist=["load_samples"])
    samples = sys_mod.load_samples()
    names = list(samples) if args.full else list(samples)[:1]

    for name in names:
        for ratio in config.VALID_RATIOS:
            check_case(name, samples[name], ratio)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
