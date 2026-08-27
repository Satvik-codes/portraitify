"""HTTP layer. Validation + JSON errors only - zero ML imports here."""
import logging
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from config import config

log = logging.getLogger("h2v.routes")
router = APIRouter(prefix="/api")

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}
MAX_DIM = 8000


def _err(status: int, code: str, message: str):
    return JSONResponse({"error": code, "message": message}, status_code=status)


def _sniff_ext(head: bytes) -> str | None:
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


@router.post("/convert")
async def convert(file: UploadFile = File(...),
                  ratio: str = Form("9:16"),
                  tier: str = Form("auto"),
                  align: str = Form("auto")):
    if ratio not in config.VALID_RATIOS:
        return _err(400, "bad_request", f"ratio must be one of {config.VALID_RATIOS}")
    if tier not in config.VALID_TIERS:
        return _err(400, "bad_request", f"tier must be one of {config.VALID_TIERS}")
    if align not in config.VALID_ALIGNS:
        return _err(400, "bad_request", f"align must be one of {config.VALID_ALIGNS}")

    raw = await file.read()
    if len(raw) == 0:
        return _err(400, "bad_request", "empty file")
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return _err(400, "bad_request",
                    f"file exceeds {config.MAX_UPLOAD_MB} MB limit")
    ext = _sniff_ext(raw[:16])
    if ext is None:
        return _err(400, "bad_request", "unsupported format (jpg/png/webp only)")

    import io
    from PIL import Image
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            w_in, h_in = im.size
            if w_in > MAX_DIM or h_in > MAX_DIM:
                return _err(400, "bad_request",
                            f"image larger than {MAX_DIM}px on a side - downscale first")
    except Exception as e:
        return _err(400, "bad_request", f"cannot decode image: {type(e).__name__}")

    from app.jobs import get_manager
    mgr = get_manager()

    tmp_path = config.UPLOADS_DIR / f".tmp-{uuid.uuid4().hex}.bin"
    tmp_path.write_bytes(raw)
    try:
        job = mgr.submit(tmp_path, {"ratio": ratio, "tier": tier, "align": align},
                         original_name=os.path.basename(file.filename or "upload"),
                         width_in=w_in, height_in=h_in)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        if type(e).__name__ == "Full":  # queue.QueueFull
            return _err(503, "busy", "server queue is full - retry shortly")
        raise
    return JSONResponse({"job_id": job.id, "status": job.status,
                         "original_url": f"/api/uploads/{job.id}"}, status_code=202)


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    from app.jobs import get_manager
    job = get_manager().get(job_id)
    if job is None:
        return _err(404, "not_found", "unknown job id")
    return job.snapshot()


@router.delete("/jobs/{job_id}")
def job_delete(job_id: str):
    from app.jobs import get_manager
    if not get_manager().delete(job_id):
        return _err(404, "not_found", "unknown job id")
    from fastapi import Response
    return Response(status_code=204)


@router.get("/uploads/{job_id}")
def upload_file(job_id: str):
    p = config.UPLOADS_DIR / f"{job_id}.bin"
    if not p.exists():
        return _err(404, "not_found", "no such upload")
    head = p.read_bytes()[:16]
    ext = _sniff_ext(head) or "png"
    media = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[ext]
    return FileResponse(p, media_type=media)


@router.get("/results/{job_id}")
def result_file(job_id: str):
    from app.jobs import get_manager
    job = get_manager().get(job_id) if job_id else None
    p = config.RESULTS_DIR / f"{job_id}.png"
    if not p.exists():
        detail = ("result not ready yet" if job and job.status in ("queued", "running")
                  else "unknown job id")
        return _err(404, "not_found", detail)
    return FileResponse(p, media_type="image/png")
