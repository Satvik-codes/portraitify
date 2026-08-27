"""Run the full auto chain on every before/after pair and render triptychs.

Produces results/compare/<name>.jpg: source | editorial after | ours, plus a
console line per pair (engine chain, engine used, wall time). Purely visual
regression harness — eyeball sign-off gates quality.
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import config
import pipeline

PAIRS = [
    ("before", "after"),
    ("before1", "after1"),
    ("before2", "after2"),
]
SAMPLES = config.PROJECT_ROOT / "tests" / "samples"
OUT = config.RESULTS_DIR / "compare"


def load(name):
    p = SAMPLES / name
    if not p.exists():
        for ext in (".jpeg", ".jpg", ".png"):
            p = SAMPLES / (name + ext)
            if p.exists():
                break
    return np.array(Image.open(p).convert("RGB"))


def triptych(src, ref, ours, out_path):
    h = 900
    cols = []
    for im in (src, ref, ours):
        w = int(im.shape[1] * h / im.shape[0])
        cols.append(np.array(Image.fromarray(im).resize((w, h))))
    total_w = sum(c.shape[1] for c in cols) + 20 * (len(cols) - 1)
    board = np.full((h + 40, total_w, 3), 24, dtype=np.uint8)
    x = 0
    for c in cols:
        board[20:20 + h, x:x + c.shape[1]] = c
        x += c.shape[1] + 20
    Image.fromarray(board).save(out_path, quality=90)


def main():
    config.ensure_dirs()
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1:] or None
    for src_name, ref_name in PAIRS:
        tag = src_name.replace("before", "")
        if only and tag not in only:
            continue
        img = load(src_name)
        ref = load(ref_name)
        t0 = time.time()
        meta = pipeline.run_smart(img, ratio="9:16", tier="auto",
                                  job_id=f"cmp{tag or '0'}")
        ours = np.array(Image.open(meta["result_path"]).convert("RGB"))
        chain = ">".join(f"{t}:{'ok' if not v else 'VOID'}"
                         for t, v in meta.get("engine_chain", []))
        print(f"{src_name}: {meta['engine']} | {chain} | "
              f"voidy={meta['void']['voidy']} cards={meta.get('cards_composed')} "
              f"| {time.time() - t0:.0f}s")
        triptych(img, ref, ours, OUT / f"cmp{tag or '0'}.jpg")
        print("   ->", OUT / f"cmp{tag or '0'}.jpg")


if __name__ == "__main__":
    main()
