"""PowerPaint 16xx NaN probe: force math SDP + vanilla attention processors.

If band black fraction ~0 -> the NaN was the known GTX-16xx mem-efficient
SDPA fp16 bug (software, fixable) and Quality tier becomes real.
If still ~1.0 -> fp16 on this GPU is genuinely unusable; Auto routes around.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import cv2
from PIL import Image

from app.jobs import _load_upload
from pipeline.placement import decide_paste
from pipeline.fillers import powerpaint_fill as pp


def main():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    print("SDP backends: math-only forced", flush=True)

    img = _load_upload(str(Path(__file__).resolve().parents[1] / "tests" / "samples" / "newsshot.png"))
    dec = decide_paste(img.shape[1], img.shape[0], 320, 576, "auto", 0.5)
    x, y, w, h = dec.paste_box
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.full((576, 320, 3), 127, np.uint8)
    canvas[y:y + h, x:x + w] = small
    mask = np.zeros((576, 320), np.uint8)
    mask[:y] = 255
    mask[y + h:] = 255

    info = torch.cuda.mem_get_info()
    print(f"VRAM free/total GB: {info[0]/2**30:.1f}/{info[1]/2**30:.1f}", flush=True)

    with pp._pipeline() as pipe:
        from diffusers.models.attention_processor import AttnProcessor
        for m in ("unet", "brushnet"):
            comp = getattr(pipe, m, None)
            if comp is None:
                continue
            try:
                comp.set_attn_processor(AttnProcessor())
                print(f"{m}: vanilla AttnProcessor set", flush=True)
            except Exception as e:
                print(f"{m}: processor swap failed: {type(e).__name__} {str(e)[:80]}", flush=True)

        g = torch.Generator("cpu").manual_seed(7)
        t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                promptA="P_ctxt", promptB="P_ctxt", promptU="",
                negative_promptA="P_obj", negative_promptB="P_obj",
                negative_promptU="",
                image=Image.fromarray(canvas), mask=Image.fromarray(mask),
                width=320, height=576, num_inference_steps=6, generator=g,
            ).images[0]
        arr = np.array(out)
        bands = np.concatenate([arr[:y], arr[y + h:]])
        black = float((bands.max(axis=2) < 12).mean())
        print(f"steps took {time.time()-t0:.0f}s | band black fraction: {black:.3f} "
              f"({'FIXED' if black < 0.05 else 'still broken'})", flush=True)
        Image.fromarray(arr).save("debug/probe_16xx_fix.png")


import time  # noqa: E402

if __name__ == "__main__":
    main()
