"""Layer detection tests against the real news-graphic sample."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from pipeline.layers import detect_layers

PASS = 0


def check(name, cond, detail=""):
    global PASS
    assert cond, f"{name} FAILED {detail}"
    PASS += 1
    print(f"ok   {name}")


def main():
    img = np.array(Image.open("tests/samples/before.jpeg").convert("RGB"))
    plan = detect_layers(img)
    kinds = {c.kind: c for c in plan.cards}

    check("graphics found", plan.found and len(plan.cards) >= 3,
          f"cards={[c.kind for c in plan.cards]}")
    check("banner detected", "banner" in kinds)
    if "banner" in kinds:
        b = kinds["banner"]
        H = img.shape[0]
        check("banner in lower half", (b.box[1] + b.box[3] / 2) / H > 0.6,
              f"cy={(b.box[1] + b.box[3] / 2) / H:.2f}")
        check("banner wide", b.box[2] >= 0.25 * img.shape[1],
              f"w={b.box[2]}")
    check("strip detected", "strip" in kinds)
    if "strip" in kinds:
        s = kinds["strip"]
        check("strip low in frame", (s.box[1] + s.box[3] / 2) / img.shape[0] > 0.7)
    check("corner logo detected", "corner_logo" in kinds)
    if "corner_logo" in kinds:
        cl = kinds["corner_logo"]
        W, H = img.shape[1], img.shape[0]
        cx, cy = (cl.box[0] + cl.box[2] / 2) / W, (cl.box[1] + cl.box[3] / 2) / H
        check("corner logo top-right", cx > 0.8 and cy < 0.25, f"({cx:.2f},{cy:.2f})")
    check("anchor cutout detected", "cutout" in kinds)
    if "cutout" in kinds:
        cu = kinds["cutout"]
        check("anchor on left", cu.box[0] <= 0.1 * img.shape[1], f"x={cu.box[0]}")
        check("anchor dominant", cu.box[2] >= 0.25 * img.shape[1] and
              cu.box[3] >= 0.5 * img.shape[0], f"{cu.box[2]}x{cu.box[3]}")
        check("cutout has real alpha", cu.alpha.max() == 255 and cu.alpha.min() < 255)

    # union mask sanity
    check("union covers cards", plan.graphic_union.max() == 255)

    # plain photo must not produce graphics (gradient synthetic)
    rng = np.random.default_rng(2)
    g = np.tile(np.linspace(30, 220, 640, dtype=np.uint8), (360, 1))
    plain = np.stack([g] * 3, axis=-1)
    plan2 = detect_layers(plain)
    check("plain photo: no graphics", not plan2.found,
          f"cards={[c.kind for c in plan2.cards]}")

    print(f"\nALL {PASS} CHECKS PASSED")


if __name__ == "__main__":
    main()
