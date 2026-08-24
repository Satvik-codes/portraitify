"""RealESRGAN polish: tiled x4 upscaling to an exact target size.

Handles the known basicsr/torchvision>=0.17 incompatibility via a shim BEFORE
any basicsr import. Used only by the Quality tier between fill and compose.
"""
import logging
import numpy as np

from config import config

log = logging.getLogger("h2v.upscale")

_shim_done = False


def _apply_basicsr_shim():
    global _shim_done
    if _shim_done:
        return
    import sys
    import types
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ImportError:
        import torchvision.transforms.functional as F
        mod = types.ModuleType("torchvision.transforms.functional_tensor")
        mod.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = mod
    _shim_done = True


def available() -> bool:
    return config.UPSCALER_PTH.exists()


def upscale_to_exact(img_rgb: np.ndarray, target_hw: tuple):
    """img_rgb HxWx3 uint8 -> uint8 at exactly target_hw=(H, W)."""
    import cv2
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w = img_rgb.shape[:2]
    scale = max(1.0, min(4.0, max(th / h, tw / w) * 1.15))

    out = _run_realesrgan(img_rgb, scale)
    if out.shape[:2] != (th, tw):
        out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    return out


def _run_realesrgan(img_rgb: np.ndarray, scale: float) -> np.ndarray:
    import torch
    from pipeline.vrac import model_slot

    def _load():
        _apply_basicsr_shim()
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        from realesrgan import RealESRGANer
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_conv=32, upscale=4, act_type="prelu")
        half = config.device() == "cuda"
        tile = 256 if half else 128
        return RealESRGANer(scale=4, model_path=str(config.UPSCALER_PTH),
                            model=model, tile=tile, tile_pad=8,
                            half=half, device=config.device())

    bgr = np.ascontiguousarray(img_rgb[..., ::-1])
    with model_slot("realesrgan", _load) as upsampler:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
        except RuntimeError as e:
            if "OutOfMemoryError" in str(type(e).__mro__) or "out of memory" in str(e).lower():
                log.warning("upscaler OOM - halving tiles and retrying once")
                upsampler.tile = max(96, upsampler.tile // 2)
                out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
            else:
                raise
    peak = ""
    try:
        import torch
        if torch.cuda.is_available():
            peak = f" peakVRAM={torch.cuda.max_memory_allocated()/2**20:.0f}MB"
    except Exception:
        pass
    log.info("upscale %sx%s x%.2f done%s", *img_rgb.shape[:2][::-1], scale, peak)
    return np.ascontiguousarray(out_bgr[..., ::-1])


if __name__ == "__main__":
    import sys
    import time
    from PIL import Image
    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    t0 = time.time()
    out = upscale_to_exact(img, (1920, 1080))
    print(f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]} "
          f"in {time.time()-t0:.2f}s")
