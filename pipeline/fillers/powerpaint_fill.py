"""Quality tier filler: PowerPaint-V2 (BrushNet) outpainting.

Artifacts must exist under third_party/ (see scripts/download_powerpaint.ps1).
Import-safe when artifacts are absent - available() gates every entry point.

NOTE: invocation kwargs are written against the Sanster/PowerPaint_v2
pipeline source; Phase 6 execution MUST re-verify against the checked-out
copy and adjust here if upstream signatures differ (see plan §10 rule 10).
"""
import logging
import time

import numpy as np

from config import config

log = logging.getLogger("h2v.powerpaint")
ENGINE = "powerpaint-v2"


def available() -> bool:
    pipe_py = (config.POWERPAINT_DIR / "powerpaint_v2" /
               "pipeline_PowerPaint_Brushnet_CA.py").exists()
    brush_dir = config.POWERPAINT_DIR / "PowerPaint_Brushnet"
    brushnet_w = brush_dir.exists() and any(brush_dir.glob("*fp16*"))
    sd_unet = (config.SD15_LOCAL_DIR / "unet" /
               "diffusion_pytorch_model.fp16.safetensors").exists()
    return bool(pipe_py and brushnet_w and sd_unet)


class QualityOOMError(RuntimeError):
    pass


def fill(canvas_rgb: np.ndarray, mask_u8: np.ndarray,
         scaled_original: np.ndarray, paste_box: tuple) -> np.ndarray:
    """Low-res semantic fill of open bands; returns full-canvas RGB."""
    import cv2
    H, W = canvas_rgb.shape[:2]

    gen_long = min(config.PP_GEN_LONG_SIDE, max(W, H))
    scale = gen_long / max(W, H)
    gw = max(320, 8 * round(W * scale / 8))
    gh = max(320, 8 * round(H * scale / 8))

    small = cv2.resize(canvas_rgb, (gw, gh), interpolation=cv2.INTER_AREA)
    mmask = cv2.resize(mask_u8, (gw, gh), interpolation=cv2.INTER_NEAREST)
    # BrushNet was trained on binary masks; feathered values degenerate its
    # conditioning into near-uniform output. Binarize; feathering happens
    # downstream in compose instead.
    mmask = np.where(mmask > 127, 255, 0).astype(np.uint8)

    with _pipeline() as pipe:
        result_small = _outpaint(pipe, small, mmask)

    filled = cv2.resize(np.asarray(result_small.convert("RGB")), (W, H),
                        interpolation=cv2.INTER_CUBIC)

    # fp16 NaN guard: BrushNet at this VRAM edge can cast NaN -> solid black
    # regions. Heal them with the FAST engine. No erosion: every black pixel
    # must be masked, or the surviving black ring becomes LaMa's context and
    # it happily continues painting black.
    for _pass in range(2):
        black = (filled.max(axis=2) < 12).astype(np.uint8)
        black[max(0, paste_box[1]):paste_box[1] + paste_box[3],
              max(0, paste_box[0]):paste_box[0] + paste_box[2]] = 0
        frac = black.mean()
        if frac < 0.005:
            break
        heal = cv2.dilate(black, np.ones((15, 15), np.uint8)) * 255
        log.info("quality NaN-heal pass %d: %.1f%% black -> lama",
                 _pass + 1, 100 * frac)
        from pipeline.fillers import lama_fill
        filled = lama_fill.fill(filled, heal)

    px, py, pw, ph = paste_box  # guarantee survives resolution change
    filled[py:py + ph, px:px + pw] = scaled_original
    return filled


# --------------------------------------------------------------------------- #

def _pipeline():
    """Context manager yielding the loaded PowerPaint pipeline inside a VRAM slot."""
    import sys
    import torch
    from contextlib import contextmanager
    from pipeline.vrac import model_slot

    @contextmanager
    def ctx():
        def _load():
            pp = str(config.POWERPAINT_DIR)
            sys.path.insert(0, pp)  # vendored modules shadow diffusers locally only
            try:
                from powerpaint_v2.BrushNet_CA import BrushNetModel
                from powerpaint_v2.pipeline_PowerPaint_Brushnet_CA import (
                    StableDiffusionPowerPaintBrushNetPipeline)
                from powerpaint_v2.power_paint_tokenizer import PowerPaintTokenizer
                from powerpaint_v2.unet_2d_condition import UNet2DConditionModel
                from transformers import CLIPTextModel, CLIPTokenizer
                from diffusers import DPMSolverMultistepScheduler

                base = str(config.SD15_LOCAL_DIR)
                # PROVEN on this machine: fp16 compute corrupts to NaN (black
                # fills) while fp32 is clean -> fp32 is the default. Set
                # FORCE_FP16=true only on verified-healthy hardware.
                dtype = torch.float16 if (config.device() == "cuda"
                                          and config.FORCE_FP16) else torch.float32
                unet = UNet2DConditionModel.from_pretrained(
                    base, subfolder="unet", variant="fp16", torch_dtype=dtype)
                brushnet = BrushNetModel.from_pretrained(
                    str(config.POWERPAINT_DIR / "PowerPaint_Brushnet"),
                    variant="fp16", torch_dtype=dtype)
                te = CLIPTextModel.from_pretrained(
                    str(config.POWERPAINT_DIR / "text_encoder_brushnet"),
                    variant="fp16", torch_dtype=dtype)
                pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                    base, unet=unet, brushnet=brushnet, text_encoder_brushnet=te,
                    torch_dtype=dtype, variant="fp16",
                    safety_checker=None, requires_safety_checker=False)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config)
                pipe.tokenizer = PowerPaintTokenizer(CLIPTokenizer.from_pretrained(
                    str(config.POWERPAINT_DIR / "tokenizer")))
                if config.device() == "cuda":
                    pipe.enable_attention_slicing()
                    pipe.enable_model_cpu_offload()  # UNet+BrushNet both heavy
                else:
                    pipe.to("cpu")
                return pipe
            finally:
                sys.path.pop(0)

        def _unload(p):
            del p

        with model_slot(ENGINE, _load, _unload) as p:
            yield p

    return ctx()


def _outpaint(pipe, image, mask):
    """Exact vendored-pipeline contract (verified against checked-out source):
    image+mask, task prompts via promptA/promptB (+negatives); no plain
    'prompt', no 'mask_image', no 'fitting_degree' on this BrushNet variant.
    """
    import torch
    from PIL import Image
    pil_img = Image.fromarray(image)
    pil_mask = Image.fromarray(mask)
    seed = int(time.time()) & 0xFFFF
    generator = torch.Generator(device="cpu").manual_seed(seed)

    try:
        with torch.inference_mode():
            return pipe(
                promptA="P_ctxt", promptB="P_ctxt",
                promptU="",  # always encoded by vendored pipeline; empty = neutral
                negative_promptA="P_obj", negative_promptB="P_obj",
                negative_promptU="",
                image=pil_img, mask=pil_mask,
                width=pil_img.size[0], height=pil_img.size[1],
                guidance_scale=5.5, num_inference_steps=config.PP_STEPS,
                generator=generator,
            ).images[0]
    except torch.cuda.OutOfMemoryError as e:
        raise QualityOOMError(str(e)) from e
