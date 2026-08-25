"""VAE fp32 upcast probe: is the NaN in the decoder or the latents?"""
import sys
import time
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
    img = _load_upload(str(Path(__file__).resolve().parents[1] / "tests" / "samples" / "newsshot.png"))
    dec = decide_paste(img.shape[1], img.shape[0], 320, 576, "auto", 0.5)
    x, y, w, h = dec.paste_box
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.full((576, 320, 3), 127, np.uint8)
    canvas[y:y + h, x:x + w] = small
    mask = np.zeros((576, 320), np.uint8)
    mask[:y] = 255
    mask[y + h:] = 255

    with pp._pipeline() as pipe:
        pipe.vae.to(torch.float32)
        print("vae dtype:", next(pipe.vae.parameters()).dtype, flush=True)
        feats = {}
        orig_dec = pipe.vae.decode
        orig_enc = pipe.vae.encode

        def spy_dec(z, *a, **k):
            feats["nan"] = bool(torch.isnan(z).any().item())
            feats["max"] = float(z.abs().max().item())
            return orig_dec(z.float(), *a, **k)

        def spy_enc(x, *a, **k):
            if isinstance(x, torch.Tensor):
                x = x.float()
            return orig_enc(x, *a, **k)

        pipe.vae.decode = spy_dec
        pipe.vae.encode = spy_enc
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
    print(f"latents NaN: {feats.get('nan')} | latent absmax: {feats.get('max'):.1f}")
    print(f"{time.time()-t0:.0f}s | band black fraction: "
          f"{float((bands.max(axis=2) < 12).mean()):.3f}")
    Image.fromarray(arr).save("debug/probe_vaefp32.png")


if __name__ == "__main__":
    main()
