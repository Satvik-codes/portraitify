"""Central configuration. Import BEFORE torch anywhere (sets TORCH_HOME)."""
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PORT = int(os.environ.get("PORT", "8000"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
FORCE_CPU = os.environ.get("FORCE_CPU", "false").lower() == "true"
FORCE_FP16 = os.environ.get("FORCE_FP16", "false").lower() == "true"
KEEP_DEBUG = os.environ.get("KEEP_DEBUG", "false").lower() == "true"

# --- Model weight paths -----------------------------------------------------
_torch_home_default = Path.home() / ".cache" / "torch"
TORCH_HOME = Path(os.environ.get("TORCH_HOME_OVERRIDE", _torch_home_default))
os.environ["TORCH_HOME"] = str(TORCH_HOME)  # must happen before torch import

LAMA_PT_PATH = TORCH_HOME / "hub" / "checkpoints" / "big-lama.pt"
YOLO_PT_PATH = PROJECT_ROOT / "models" / "yolov8n-seg.pt"
UPSCALER_PTH = PROJECT_ROOT / "models" / "realesr-general-x4v3.pth"
POWERPAINT_DIR = PROJECT_ROOT / "third_party" / "PowerPaint_v2"
SD15_BASE_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_LOCAL_DIR = PROJECT_ROOT / "third_party" / "sd15_base"

# --- Canvas presets ----------------------------------------------------------
CANVAS_SIZES = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
}
VALID_RATIOS = tuple(CANVAS_SIZES.keys())
VALID_TIERS = ("auto", "fast", "quality", "sdturbo", "relayout")
VALID_ALIGNS = ("auto", "top", "center", "bottom")

# --- Pipeline tuning ---------------------------------------------------------
FEATHER_PX = 40
OVERLAP_PX = 48
RAMP_PX = 16
PP_GEN_LONG_SIDE = 576   # throttled-GPU budget: (576/1120)^2 ~= 3.8x fewer px
PP_STEPS = 14            # DPM++ multistep; tuned for <=~2.5 min fill on GTX 1650
YOLO_PERSON_CONF = 0.35
YOLO_ANY_CONF = 0.45
AUTO_CENTROID_TARGET = 0.42  # subject centroid maps here on auto align

# --- Runtime dirs ------------------------------------------------------------
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
DEBUG_DIR = PROJECT_ROOT / "debug"
RUNTIME_DIRS = (UPLOADS_DIR, RESULTS_DIR, LOGS_DIR, DEBUG_DIR)

RESULT_TTL_HOURS = 24
RESULT_KEEP_COUNT = 100


def ensure_dirs():
    for d in RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def device() -> str:
    if FORCE_CPU:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"name": props.name, "vram_gb": round(props.total_memory / 1024**3, 1),
                    "cuda": torch.version.cuda}
    except Exception as e:
        return {"error": str(e)}
    return None
