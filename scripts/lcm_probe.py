"""SD1.5-inpaint + LCM-LoRA probe: fp16 NaN check + timing on before.jpeg.

Decides the smart engine's fill dtype/speed (see implementation-plan-v3.md §7).
"""
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
from config import config


def snap(repo):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


def main():
    dtype_name = sys.argv[1] if len(sys.argv) > 1 else "fp16"
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32

    img = _load_upload(str(Path(__file__).resolve().parents[1] / "tests" / "samples" / "before.jpeg"))
    dec = decide_paste(img.shape[1], img.shape[0], 320, 576, "auto", 0.5)
    x, y, w, h = dec.paste_box
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.full((576, 320, 3), 127, np.uint8)
    canvas[y:y + h, x:x + w] = small
    mask = np.zeros((576, 320), np.uint8)
    mask[:y] = 255
    mask[y + h:] = 255

    from diffusers import AutoPipelineForInpainting, LCMScheduler
    t0 = time.time()
    pipe = AutoPipelineForInpainting.from_pretrained(
        snap("botp/stable-diffusion-v1-5-inpainting"), torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    pipe.load_lora_weights(snap("latent-consistency/lcm-lora-sdv1-5"))
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"loaded in {time.time()-t0:.0f}s | dtype={dtype} | "
          f"VRAM {torch.cuda.memory_allocated()/2**30:.1f}GB", flush=True)

    g = torch.Generator("cpu").manual_seed(11)
    t0 = time.time()
    with torch.inference_mode():
        out = pipe(prompt="news photograph, night crowd, street lights, sharp",
                   image=Image.fromarray(canvas), mask_image=Image.fromarray(mask),
                   width=320, height=576, num_inference_steps=8,
                   guidance_scale=1.0, generator=g).images[0]
    dt = time.time() - t0
    arr = np.array(out)
    bands = np.concatenate([arr[:y], arr[y + h:]])
    black = float((bands.max(axis=2) < 12).mean())
    print(f"{dtype_name}: {dt:.0f}s for 8 steps | band black fraction: {black:.3f} "
          f"({'FP16 CLEAN - USE IT' if black < 0.05 and dtype_name=='fp16' else 'verdict above'})",
          flush=True)
    Image.fromarray(arr).save(f"debug/probe_lcm_{dtype_name}.png")


if __name__ == "__main__":
    main()
