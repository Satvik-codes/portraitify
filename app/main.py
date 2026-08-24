"""FastAPI entry point. Import config first (sets TORCH_HOME before torch)."""
from config import config
config.ensure_dirs()

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / "app.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("h2v")

from app.routes import router  # noqa: E402
from app.jobs import get_manager  # noqa: E402

app = FastAPI(title="H2V Smart Converter", docs_url=None, redoc_url=None)
app.include_router(router)


@app.get("/healthz")
def healthz():
    from pipeline.fillers import lama_fill
    try:
        from pipeline.fillers import powerpaint_fill
        quality_ok = powerpaint_fill.available()
    except Exception:
        quality_ok = False
    info = config.gpu_info() or {}
    return {
        "ok": True,
        "gpu": info.get("name"),
        "vram_gb": info.get("vram_gb"),
        "torch_cuda": bool(info.get("cuda")),
        "fast_tier": lama_fill.available(),
        "quality_tier": quality_ok,
        "ratios": list(config.VALID_RATIOS),
        "queue_max": get_manager().QUEUE_MAX,
    }


@app.get("/")
def index():
    index_html = Path(__file__).parent / "static" / "index.html"
    if index_html.exists():
        return FileResponse(index_html, media_type="text/html")
    return {"ok": True, "service": "H2V Smart Converter", "ui": "pending Phase 9"}


@app.on_event("startup")
def _startup():
    get_manager()  # boots worker + recovery/purge
    log.info("H2V up on http://127.0.0.1:%s", config.PORT)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=config.PORT)
