"""Smart engine guarantees: card fidelity + photo region + void gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2
from PIL import Image

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"{name} FAILED {detail}"
    PASS += 1
    print(f"ok   {name}")


def main():
    from app.jobs import _load_upload
    import pipeline

    img = _load_upload("tests/samples/before.jpeg")
    meta = pipeline.run_smart(img, ratio="9:16", tier="auto",
                              job_id="fidelity-test")
    check("engine is smart", str(meta["engine"]).startswith("smart"),
          f"engine={meta['engine']}")
    check("cards composed", (meta.get("cards_composed") or 0) >= 3,
          f"cards={meta.get('cards_composed')}")
    check("fidelity ok", meta.get("fidelity_ok") is True)
    check("not voidy", not meta["void"]["voidy"], str(meta["void"]))

    result = np.array(Image.open(meta["result_path"]).convert("RGB"))
    check("output size", result.shape[:2] == (1920, 1080),
          str(result.shape))

    # photo band check: rebuild the smart zoom geometry from meta
    px, py, pw, ph = meta["paste_box"]
    crop_x = int(meta.get("smart_crop_x", 0))
    interp = cv2.INTER_AREA if (pw * ph) < img.shape[0] * img.shape[1] else cv2.INTER_CUBIC
    scaled_full = cv2.resize(img, (int(round(img.shape[1] * ph / img.shape[0])), ph),
                             interpolation=interp)
    band = scaled_full[:, crop_x:crop_x + pw]
    region = result[py:py + ph, px:px + pw]
    matches = int((region == band).all(axis=2).sum())
    total = pw * ph
    # Cleanup restores originals outside card holes; differences vs raw zoom
    # are localized to the fill's 48px context ring + card overlaps (~30%).
    # Strict byte-exactness for plain tiers lives in test_pixel_guarantee.
    check("photo band >=65% byte-exact vs original zoom",
          matches / total >= 0.65, f"{matches}/{total}")

    print(f"\nALL {PASS} CHECKS PASSED")


if __name__ == "__main__":
    main()
