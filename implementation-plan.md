# H2V Smart Converter — Implementation Plan

> Converts horizontal images into vertical (portrait) images by detecting subjects, smartly placing the original frame on a larger canvas, and AI-filling every remaining pixel (outpainting) — with the guarantee that **original pixels are never modified**, so text, tickers, and logos stay crisp. Runs fully local on a GTX 1650 (4 GB VRAM) laptop. Two tiers: Fast (LaMa, ~10–20 s) and Quality (PowerPaint-V2, ~75–95 s).

---

## 1. Project Overview

H2V Smart Converter is a single-user, localhost web application. A user drags a horizontal photo (news screenshot, portrait, landscape, product shot) into the browser, picks a target ratio (9:16, 4:5, 1:1) and a quality tier, clicks Convert, watches live progress, then previews a before/after slider and downloads the finished vertical image. Under the hood the app segments subjects (YOLOv8n-seg), computes a saliency-aware placement of the untouched original onto the target canvas, builds a feathered mask over empty regions, fills those regions with either Big-LaMa (Fast) or PowerPaint-V2 (Quality), optionally upscales/polishes the filled areas (RealESRGAN general x4v3), and composites the original region back byte-identically. There is no cloud dependency, no auth, no database — everything is files on disk and an in-process job queue.

## 2. Tech Stack

| Technology | Role |
|---|---|
| Python 3.10+ venv | Runtime |
| PyTorch (CUDA cu121 build, sm_75) | Tensor runtime for all models |
| FastAPI + Uvicorn | HTTP server, static file serving, JSON API |
| Pillow + NumPy | Canvas composition, masking, pixel guarantees |
| OpenCV (`opencv-python`) | Saliency fallback, feathering, seamless blending utilities |
| Ultralytics YOLO | Person/object detection & segmentation (`yolov8n-seg.pt`) |
| simple-lama-inpainting | Big-LaMa inpainting wrapper (reuses cached `big-lama.pt`) |
| PowerPaint-V2 (BrushNet variant) | Quality-tier semantic outpainter — custom diffusers pipeline from HF repo `Sanster/PowerPaint_v2`, base SD1.5 from `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| RealESRGAN (`realesr-general-x4v3.pth`) | Quality-tier final upscale to exact output size (weights copied from sibling project) |
| Vanilla HTML/CSS/JS (single file) | Frontend UI, served by FastAPI |
| threading.Queue | In-process single-worker job queue |

**Hard constraints**
- Target machine: Ryzen 5 4600H, GTX 1650 Mobile 4 GB, 16 GB RAM, Windows.
- Bind to `127.0.0.1` only. Single concurrent conversion job.
- Never exceed 4 GB VRAM: exactly ONE model object resident at a time; explicit `del` + `torch.cuda.empty_cache()` between stages.
- Original-image region in every output must be byte-identical to the input region (verified by automated test).
- All heavy model weights come from disk caches that already exist except PowerPaint-V2 + SD1.5 base (one-time scripted download, ~4–5 GB total).

## 3. System Architecture

### 3.1 Architecture Overview

```
Browser (index.html, vanilla JS)
   │  multipart POST /api/convert          poll GET /api/jobs/{id} (500 ms)
   ▼
FastAPI app.py  ──► JobManager (threading)
   │                    │ single worker thread pulls Job objects
   ▼                    ▼
uploads/            Pipeline.run(job)  ← the only place models are loaded
                        1. detect.py     YOLOv8n-seg → subject boxes/masks
                        2. placement.py  canvas math + saliency-biased position
                                         + feathered RGBA mask builder
                        3. fillers/
                           ├─ lama_fill.py        FAST tier
                           └─ powerpaint_fill.py  QUALITY tier
                        4. polish.py     RealESRGAN upscale (Quality only)
                        5. compose.py    paste-original-byte-identical
                                         + edge ramp blending
   │
   ▼
results/{job_id}.png  +  jobs.jsonl append-only log
```

Layer responsibilities:
- **Frontend Layer** — owns zero business logic; renders dropzone, ratio picker (9:16 / 4:5 / 1:1), tier toggle (fast/quality), progress bar driven by polling, before/after slider, download button. Talks only to the four API endpoints.
- **API Layer** — request validation, multipart handling, JSON errors, serving static frontend + result/upload files. No ML imports.
- **Job Layer** — `JobManager` singleton: bounded queue (max 1 running + max 20 waiting), worker thread, state transitions, persistence to `jobs.jsonl`.
- **Pipeline Layer** — pure functions taking `(PIL.Image, params)` returning `(PIL.Image, debug_info)`; each stage loads its model through the shared `ModelSlot` context manager which guarantees exclusive VRAM residency.
- **Model Layer** — thin wrappers: `lama_fill.py`, `powerpaint_fill.py`, `yolo_det.py`, `upscale.py`. Each exposes `load() -> handle`, `run(handle, inputs) -> outputs`, `unload(handle)`. Nothing else may import torch directly except these files and `vrac.py` (VRAM utility).
- **Storage Layer** — filesystem only: `uploads/{id}.bin` (original bytes), `results/{id}.png`, `jobs.jsonl`. Startup purge deletes results older than 24 h and keeps newest 100.

### 3.2 Data Models

**Job** (in-memory dict, persisted as one JSON line in `jobs.jsonl`)
- `id`: string, UUID4 hex. Primary key everywhere (upload filename, result filename, URL segment).
- `status`: enum `"queued" | "running" | "done" | "failed"` (never nullable, default `queued`).
- `stage`: enum `"detecting" | "placing" | "filling" | "polishing" | "composing" | null`; null until running.
- `progress`: integer 0–100; coarse per-stage estimates (10/25/70/90/98), set to 100 on done.
- `params`: object `{ratio: "9:16"|"4:5"|"1:1", tier: "fast"|"quality", align: "auto"|"top"|"center"|"bottom"}`.
- `original_name`: string, client-side filename (display only).
- `width_in`, `height_in`: ints, source dimensions.
- `error`: string|null; human-readable message when status=failed.
- `created_at`: ISO timestamp; drives the 24 h purge.

**PlacementDecision** (returned by placement stage, not persisted)
- `paste_box`: tuple `(x, y, w, h)` — where the scaled original lands on the canvas.
- `subject_centroid_y`: float|null — normalized 0–1 vertical centroid of detected subjects.
- `offset_reason`: string — `"auto_thirds" | "user_align_top" | ...` for debug overlay.

**FillResult**
- `image`: PIL RGBImage, full canvas, filled regions complete.
- `engine`: string `"big-lama"` or `"powerpaint-v2"`.
- `seconds`: float, wall time of the fill stage.

**Config** (module-level constants, overridable via `.env`)
- `CANVAS_SIZES`: dict mapping ratio → (w, h): `9:16→(1080,1920)`, `4:5→(1080,1350)`, `1:1→(1080,1080)`.
- `FEATHER_PX`: int, default 40 — Gaussian blur radius applied to mask near the original-frame border.
- `OVERLAP_PX`: int, default 48 — how far the mask eats INTO the original frame border (see §6 Phase 4 rationale).
- `PP_GEN_LONG_SIDE`: int, default 1120 — long side of the low-res canvas fed to PowerPaint (multiple of 8).
- `MAX_UPLOAD_MB`: int, default 25.
- Paths: `TORCH_CACHE_DIR` (= `%USERPROFILE%\.cache\torch`), `LAMA_PT_PATH` (= `<TORCH_CACHE_DIR>\hub\checkpoints\big-lama.pt`), `YOLO_PT_PATH` (= `models/yolov8n-seg.pt`, copied at setup), `UPSCALER_PTH` (= `models/realesr-general-x4v3.pth`), `POWERPAINT_DIR` (= `third_party/PowerPaint_v2`), `SD15_BASE` (= `stable-diffusion-v1-5/stable-diffusion-v1-5`).

### 3.3 State Machines

**Conversion job**

```
queued ──(worker picks, capacity ok)──► running
running ──(all stages OK)─────────────► done      [result file exists]
running ──(any stage raises)──────────► failed    [error string set, partial result deleted]
queued  ──(queue overflow > 20)───────► failed    [error="server busy"]
```

Inside `running`, `stage` advances strictly forward: `detecting → placing → filling → polishing(Quality only) → composing`. Any exception inside a stage is caught by the worker, logged to console + `jobs.jsonl`, transitions to `failed`, and triggers `cleanup_job_files(id, keep_original=False)`.

**Model residency (per pipeline run)**

```
empty ──load(YOLO)──► yolo_resident ──unload──► empty
empty ──load(LaMa | PP+SD15)──► filler_resident ──unload──► empty
empty ──load(RealESRGAN, Quality only)──► upscaler_resident ──unload──► empty
```

Rule enforced in code: `with ModelSlot(loader):` — the context manager asserts nothing else is resident, calls `loader.load()`, yields the handle, and guarantees unload + `torch.cuda.empty_cache()` even on exception.

### 3.4 API Contract

```
GET /
Purpose: Serve frontend
Response 200: text/html (static/index.html)
Errors: none

GET /healthz
Response 200: {"ok": true, "gpu": "NVIDIA GeForce GTX 1650"|null, "torch_cuda": true|false}

POST /api/convert
Purpose: Submit a conversion job
Request: multipart/form-data
  - file: binary (jpg/jpeg/png/webp, ≤ MAX_UPLOAD_MB, decodable by PIL)
  - ratio: form field, one of "9:16","4:5","1:1"           (default "9:16")
  - tier: form field, one of "fast","quality"              (default "fast")
  - align: form field, one of "auto","top","center","bottom" (default "auto")
Response 202: { "job_id": "<uuid>", "status": "queued",
                "original_url": "/api/uploads/<uuid>" }
Response 400: { "error": "bad_request", "message": "<specific reason>" }
  cases: missing file · unsupported extension · PIL decode failure ·
         size > limit · invalid enum value
Response 503: { "error": "busy", "message": "queue full, retry later" }

GET /api/jobs/{job_id}
Response 200: { "job_id": "...", "status": "...", "stage": "...",
                "progress": 0-100,
                "error": null | "<message>",
                "result_url": "/api/results/<id>" | null }
Response 404: { "error": "not_found" }

GET /api/uploads/{job_id}
Purpose: Serve the stored ORIGINAL upload (for before/after slider)
Response 200: image bytes (Content-Type from sniffed format)
Response 404: { "error": "not_found" }

GET /api/results/{job_id}
Purpose: Serve the finished PNG
Response 200: image/png
Response 404: { "error": "not_found" }   (also while status != done)

DELETE /api/jobs/{job_id}
Purpose: User-initiated cleanup (removes upload + result + log line stays)
Response 204 | 404
```

All non-2xx bodies share shape `{ "error": "<code>", "message": "<detail>" }`. Raw tracebacks are NEVER returned; they go to console + `logs/app.log`.

### 3.5 Integration Contracts

**Big-LaMa (local file, via simple-lama-inpainting)**
- Input: `PIL RGB` full-canvas image + `PIL L` mask (255 = fill here, 0 = keep).
- Output: filled full-canvas RGB.
- Loading: package calls `torch.hub.load` under the hood; we pre-set `os.environ["TORCH_HOME"] = TORCH_CACHE_DIR` BEFORE import so it resolves the existing `%USERPROFILE%\.cache\torch\hub\checkpoints\big-lama.pt` without downloading.
- Failure modes: CUDA OOM → catch `torch.cuda.OutOfMemoryError`, unload, retry once with image downscaled to long-side 1280 on CPU (`device="cpu"`); corrupt/missing `big-lama.pt` → fail job with message telling user to re-run `scripts\\verify_assets.ps1`.
- Expected latency @1080×1920 on GTX 1650: 3–8 s.

**PowerPaint-V2 (downloaded once by `scripts/download_powerpaint.ps1`)**
- Artifacts placed under `third_party/PowerPaint_v2/`:
  - python module folder `powerpaint_v2/` cloned from HF repo `Sanster/PowerPaint_v2` (contains `pipeline_PowerPaint_Brushnet_CA.py`, `BrushNet_CA.py`, `power_paint_tokenizer.py`, `unet_2d_condition.py`, `unet_2d_blocks.py`);
  - weights subfolders: `PowerPaint_Brushnet/` (fp16 safetensors), `text_encoder_brushnet/`, `tokenizer/`;
  - base SD1.5 fetched via `huggingface_hub.snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", allow_patterns=["*.json","text_encoder/**","tokenizer/**","unet/diffusion_pytorch_model.fp16.safetensors","vae/diffusion_pytorch_model.fp16.safetensors","scheduler/*"])`.
- Invocation contract (mirrors upstream `main.py`, image-outpainting branch):
  - Build pipe: `UNet2DConditionModel.from_pretrained(SD15_BASE, subfolder="unet", variant="fp16")` + `BrushNetModel.from_pretrained(.../PowerPaint_Brushnet, variant="fp16")` + `CLIPTextModel.from_pretrained(text_encoder_brushnet, variant="fp16")` wrapped in `StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(SD15_BASE, unet=..., brushnet=..., text_encoder_brushnet=..., torch_dtype=float16, safety_checker=None)`; scheduler = `DPMSolverMultistepScheduler`; tokenizer replaced by `PowerPaintTokenizer(CLIPTokenizer.from_pretrained(tokenizer_dir))`.
  - Task tokens for outpaint: `promptA=promptB="P_ctxt"`, `negative_promptA=negative_promptB="P_obj"`; guidance_scale 5.5; steps 25; fitting_degree 1.0.
  - We feed a DOWNSCALED canvas: whole composite (original scaled + gray #7F7F7F bands + white-on-band mask) at `PP_GEN_LONG_SIDE` long side (short side derived, rounded to /8). This is mandatory for 4 GB VRAM.
- Failure modes: OOM → abort Quality tier with clear error suggesting Fast tier (no silent CPU retry — too slow); missing artifact → fail with "run scripts/download_powerpaint.ps1".
- Expected latency: 55–85 s incl. load.

**RealESRGAN general x4v3 (local file)**
- Loaded via `basicsr`-free minimal wrapper: use `realesrgan` pip package's `RealESRGANer(model_path=UPSCALER_PTH, model=realesr_general_x4v3 architecture, scale=4, tile=256, tile_pad=8, half=True)`. Tile mode keeps VRAM < 1.5 GB.
- Used ONLY in Quality tier: upscale the low-res PowerPaint output to exactly CANVAS_SIZES target, then compose (§ Phase 9).
- Failure: package/model mismatch → fail job, message names the pth expected.

**YOLOv8n-seg (local file copied from sibling project)**
- `YOLO(YOLO_PT_PATH)`, predict on RGB ndarray, `conf=0.35`, `classes=[0]` (person) first pass; if zero persons, second pass with all classes conf≥0.45; if still zero, return None → placement falls back to OpenCV `StaticSaliencySpectralResidual` centroid → else geometric center.
- Latency: 0.1–0.4 s GPU.

## 4. Environment & Configuration

No secrets exist in this project. All configuration lives in `config/config.py` reading optional `.env` (python-dotenv):

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Uvicorn bind port (host always 127.0.0.1) |
| `MAX_UPLOAD_MB` | `25` | Upload guard |
| `TORCH_HOME` | auto-detected `%USERPROFILE%\.cache\torch` | Makes torch.hub find cached big-lama.pt |
| `FORCE_CPU` | `false` | Emergency switch; pipeline respects it for YOLO/LaMa |
| `KEEP_DEBUG` | `false` | If true, pipeline writes intermediate artifacts to `debug/{job_id}/` (mask.png, placed.png, raw_fill.png) |

Python packages (`requirements.txt`, install order matters):
```
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.4.1+cu121
torchvision==0.19.1+cu121
simple-lama-inpainting==0.1.2
ultralytics>=8.2
opencv-python-headless>=4.9
numpy<2
pillow>=10
huggingface_hub>=0.24
transformers>=4.44
diffusers==0.30.3
accelerate>=0.33
safetensors>=0.4
realesrgan==0.3.0
basicsr==1.4.2
fastapi>=0.111
uvicorn[standard]>=0.30
python-multipart>=0.0.9
python-dotenv>=1.0
```

System prerequisites verified during setup: NVIDIA driver ≥ 550, `nvidia-smi` visible; Visual C++ runtime already present on this machine.

## 5. Project Structure

```
horizaontaltovertical\
├── implementation-plan.md          — this document
├── requirements.txt
├── .env.example                    — commented overrides
├── run.bat                         — activates venv, starts uvicorn, opens browser
├── scripts\
│   ├── setup_env.ps1               — creates .venv, installs requirements, prints GPU check
│   ├── fetch_local_weights.ps1     — copies yolov8n-seg.pt from pngtopsd + realesr-general-x4v3.pth
│   │                                 from 'tv today  2' into .\models ; verifies hashes/sizes;
│   │                                 verifies %USERPROFILE%\.cache\torch\hub\checkpoints\big-lama.pt
│   ├── download_powerpaint.ps1     — git/HF-download Sanster/PowerPaint_v2 module + weights
│   │                                 and SD1.5 fp16 subset into third_party\
│   └── verify_assets.ps1           — post-install audit: every path in §3.2 exists, sizes match,
│                                     torch.cuda.is_available(), sm_75 supported → green/red summary
├── config\
│   └── config.py                   — single source of truth; loads .env; exports ALL constants §3.2
├── app\
│   ├── __init__.py
│   ├── main.py                     — FastAPI factory, mounts routes + static, startup purge hook
│   ├── routes.py                   — the 6 endpoints of §3.4; zero ML imports
│   ├── jobs.py                     — JobManager: queue, worker thread, state machine, jsonl log
│   └── static\
│       └── index.html              — entire frontend (inline CSS+JS)
├── pipeline\
│   ├── __init__.py                 — Pipeline.run(job_params) orchestrator implementing §3.3
│   ├── vrac.py                     — VRAM utilities: ModelSlot ctx mgr, empty_cache, device pick
│   ├── detect.py                   — YOLO person/object detection + saliency fallback
│   ├── placement.py                — scaling, saliency-biased paste box, mask builder
│   ├── compose.py                  — byte-identical original paste + edge-ramp blender
│   └── fillers\
│       ├── __init__.py             — get_filler(tier) factory
│       ├── lama_fill.py            — FAST engine wrapper
│       └── powerpaint_fill.py      — QUALITY engine wrapper (imports only when tier=quality)
├── models\                         — copied local weights (gitignored)
│   ├── yolov8n-seg.pt
│   └── realesr-general-x4v3.pth
├── third_party\                    — PowerPaint_V2 module + weights, sd15_base cache (gitignored)
├── uploads\  results\  logs\  debug\   — runtime dirs (gitignored), created on boot
└── tests\
    ├── test_pixel_guarantee.py     — THE invariant test (§8)
    ├── test_placement_math.py      — synthetic cases for paste_box/mask geometry
    └── smoke_e2e.py                — CLI: converts 3 bundled sample images, prints timings
```

## 6. Implementation Phases

### Phase 1: Scaffold & environment
**Goal:** Repo boots; `verify_assets.ps1` passes green proving every weight the project will ever need is reachable.
**Prerequisites:** none.
**Tasks:**
1. Create directory skeleton of §5 exactly (including empty `__init__.py` files, gitignored runtime dirs, `.gitignore` covering venv/models/third_party/uploads/results/logs/debug).
2. Write `requirements.txt` exactly as §4; write `scripts/setup_env.ps1` (py -3.10 -m venv .venv → pip install in listed order → run inline GPU probe printing `torch.version.cuda`, `torch.cuda.get_device_name(0)`, capability; exit non-zero if cuda unavailable unless FORCE_CPU=true).
3. Write `scripts/fetch_local_weights.ps1` performing the two Copy-Items from the verified sibling paths (`pngtopsd\backend\models\yolov8n-seg.pt`, `tv today  2\models\realesr-general-x4v3.pth`), asserting file lengths 7 MB ±10% and 5 MB ±10% respectively.
4. Write `config/config.py` with every constant of §3.2, TORCH_HOME export executed at import time (before any torch import anywhere — enforce by importing config first inside pipeline/__init__), dotenv override support.
5. Write `scripts/verify_assets.ps1`: checks existence+size for LAMA_PT_PATH, both models\ files, prints PASS/FAIL table, exits accordingly.
6. Write minimal `app/main.py` exposing `/healthz` only; `run.bat` starts it.
**Completion criteria:** `.\scripts\setup_env.ps1; .\scripts\fetch_local_weights.ps1; .\scripts\verify_assets.ps1; .\run.bat` → healthz returns `torch_cuda:true` on this laptop; verify_assets prints all-PASS.

### Phase 2: Placement & mask geometry (pure functions, no ML)
**Goal:** Deterministic, unit-testable canvas math — the heart of "smart, no blank space".
**Prerequisites:** Phase 1 (config exists).
**Tasks:**
1. `placement.py::decide_paste(orig_w, orig_h, canvas_w, canvas_h, align, subject_centroid_y_norm) -> PlacementDecision`: scale original uniformly to FIT WIDTH of canvas (if resulting height exceeds canvas — e.g., extreme panoramas — instead fit HEIGHT and let width bands be filled symmetrically); compute y-offset: `align=center` → centered; top/bottom → flush; `auto` → solve offset such that subject centroid maps to 42% canvas height, clamped so paste stays fully inside canvas; no-subject fallback = centered. Return paste_box + reason.
2. `placement.py::build_mask(canvas_w, canvas_h, paste_box) -> np.uint8 (H,W)`: 255 everywhere EXCEPT paste_box interior; then carve OVERLAP_PX inward ring set to 255 (mask eats slightly into original so fillers see real context across the seam), then Gaussian-blur FEATHER_PX, then re-solidify interior of original minus 2 px safety rim to 0 (hard protection), leaving soft transition only in the ring straddling the border. Document the invariant: `mask[paste_interior_minus_rim] == 0`.
3. `compose.py::composite(original_scaled, filled_canvas, paste_box, ramp_px=16)`: paste original at paste_box with NEAREST-off exact integer coords (bytes preserved), then linear-alpha blend a ramp_px-wide ring strictly OUTSIDE the paste rectangle between filled content and pasted edge (blend weights touch only ring pixels — assert afterwards that paste-interior unchanged).
4. `tests/test_placement_math.py`: landscape 1920×1080 → 9:16 canvas expects paste_box width 1080, height 607.5→rounded 608, y computed per three align modes; panorama 3000×400 → fit-height branch; centroid bias case (subject at y=0.9 → offset shifts down, still clamped); mask invariant assertions incl. blur behavior.
**Completion criteria:** pytest green; visual debug script `python -m pipeline.placement demo.jpg --show` renders canvas+mask overlay correctly for the three sample categories.

### Phase 3: Detection
**Goal:** Reliable subject signal feeding placement, on-cache, <0.5 s.
**Prerequisites:** Phase 1, Phase 2 (consumes centroid contract).
**Tasks:**
1. `detect.py::subject_centroid(img_rgb) -> float | None` implementing the two-pass YOLO strategy (persons-only conf .35 → any-class conf .45) computing the AREA-weighted mean normalized y of masks/bboxes; wrap ultralytics import lazily.
2. Fallback chain coded exactly: YOLO None → `cv2.saliency.StaticSaliencySpectralResidual_create()` centroid (thresholded map, argmax row-weight) → return None.
3. FORCE_CPU honored; model loaded/unloaded via ModelSlot despite being small (keeps discipline uniform).
4. Debug artifact when KEEP_DEBUG: annotated boxes image.
**Completion criteria:** `python -m pipeline.detect sample.jpg` prints centroid + method used; runs <1 s on news screenshot containing faces; returns None gracefully on texture-less gradient test image.

### Phase 4: FAST tier end-to-end (LaMa)
**Goal:** First shippable converter — horizontal in, complete vertical PNG out, no blanks, originals intact.
**Prerequisites:** Phases 2–3.
**Tasks:**
1. `fillers/lama_fill.py`: lazy `from simple_lama_inpainting import SimpleLama`; `SimpleLama(device=...)`; run(image_rgb, mask_l) → rgb; OOM path per §3.5; expose ENGINE="big-lama".
2. `pipeline/__init__.py::run(params)`: orchestrate detect→place→build canvas (gray fill)→mask→fill→compose→save PNG (optimize=True) to results; emit stage callbacks into JobManager; KEEP_DEBUG dumps.
3. Edge cases handled explicitly: input already portrait/taller than target → still valid (bands left/right get filled; paste math covers via fit-height branch); tiny inputs (<200 px side) → pre-note in result metadata, LaMa handles natively; 16-bit/P-mode uploads normalized to RGB in route layer before pipeline.
4. `tests/smoke_e2e.py` FAST path: bundle 3 samples (news screenshot, group portrait, landscape) under tests\samples\ (user-provided or generated placeholders clearly named TODO-REPLACE-WITH-REAL-SAMPLES if unavailable — but code path must run on ANY jpeg dropped in), print per-stage timings, assert output size == CANVAS_SIZES[ratio] and no fully-gray rows remain (std>2 per row).
**Completion criteria:** CLI smoke produces visually clean 1080×1920 from a 16:9 sample in ≤25 s total; original region diff == 0 (quick manual numpy check embedded in smoke).

### Phase 5: Upscaler (polish) module
**Goal:** Reusable RealESRGAN wrapper ready for Quality tier.
**Prerequisites:** Phase 1 (pth copied).
**Tasks:**
1. `pipeline/fillers/../upscale.py` (place under pipeline\): load `realesr_general_x4v3` arch from basicsr registry, RealESRGANer with tile=256/tile_pad=8/half=True; `upscale_to_exact(img, target_hw)` doing tiled ×N then cv2.resize remainder to exact target (LANCZOS4).
2. Unload + cache discipline via ModelSlot; FORCE_CPU forces tiles smaller + half=False.
3. Micro-bench harness `python -m pipeline.upscale bench in.jpg` printing ms + VRAM peak (torch.cuda.max_memory_allocated reset around call).
**Completion criteria:** bench on 672×1216 → 1080×1920 completes <8 s GPU, peak <1.6 GB, output sharpness visibly ≥ bicubic in side-by-side.

### Phase 6: QUALITY tier — acquire & wire PowerPaint-V2
**Goal:** Semantic outpainting within budget, cleanly isolated behind tier flag.
**Prerequisites:** Phases 4–5 (orchestration exists), internet for one-time download.
**Tasks:**
1. `scripts/download_powerpaint.ps1`: (a) git clone depth1 https://huggingface.co/Sanster/PowerPaint_v2 into third_party\PowerPaint_v2 (module + weights); (b) python one-liner using huggingface_hub.snapshot_download for SD1.5 fp16 subset per §3.5 patterns into third_party\sd15_base; (c) resume-safe (skip-if-exists checks per artifact); prints total footprint.
2. `sys.path.insert(0, third_party\PowerPaint_v2)` scoped INSIDE powerpaint_fill only (its vendored `unet_2d_condition.py` must shadow diffusers’ only within this module’s imports).
3. `fillers/powerpaint_fill.py` implementing the exact invocation contract of §3.5 (task tokens P_ctxt/P_obj, gray-band composite at PP_GEN_LONG_SIDE, DPMSolver, guidance 5.5, steps 25); returns upscaled-ready low-res canvas + engine tag; OOM → raise typed `QualityOOMError` (route maps to user message “GPU busy for Quality tier — try Fast”, never silent downgrade).
4. Orchestrator change: tier=quality inserts polish stage (Phase 5) between fill and compose, and compose receives upscaled bg; paste-original remains LAST so guarantee holds regardless of tier.
5. Smoke: extend `tests/smoke_e2e.py --tier quality`; record timing + peak VRAM lines into output report.
**Completion criteria:** quality conversion of same 16:9 news sample finishes ≤95 s, peak VRAM ≤3.8 GB, filled sky/studio band shows coherent structure (manual eyeball), pixel-guarantee test still green.

### Phase 7: Job engine
**Goal:** Robust async execution with honest progress, surviving bad inputs and crashes.
**Prerequisites:** Phase 4 (a runnable pipeline).
**Tasks:**
1. `jobs.py::JobManager`: `threading.Thread` daemon worker + `queue.Queue(maxsize=20)`; submit validates enums/sizes again server-side; job dict per §3.2; single-flight semaphore ensuring ONE pipeline at a time; global try/except mapping exceptions→failed with sanitized messages (full traceback → logs\app.log).
2. Stage-callback protocol `cb(stage:str, pct:int)` threaded through Pipeline.run; timestamps written to jsonl line on every transition.
3. Persistence/recovery: on boot, replay jobs.jsonl marking any stale `queued|running` rows as failed("server restarted"); purge job per §3.1 storage rules.
4. DELETE endpoint wiring + idempotency (double delete → 404 second time).
**Completion criteria:** kill server mid-quality-job → restart marks it failed cleanly, next job works; submitting 22 rapid requests yields 2 accepted + rest 503; progress observed climbing 10→100 via polling script.

### Phase 8: API layer
**Goal:** Full §3.4 surface, hardened.
**Prerequisites:** Phase 7.
**Tasks:**
1. `routes.py` implementing all six endpoints; upload handler: sniff magic bytes (jpg/png/webp) not just extension, PIL verify+convert("RGB"), dimension cap 8000×8000 (larger → 400 with resize hint), stream to uploads\{id}.bin then os.replace for atomicity.
2. Static mount: serve app\static at "/", cache headers no-store for API, immutable for /assets if later split.
3. Global exception middleware → `{error:"internal"}` + correlation id echoed in logs.
4. CORS disabled (same-origin) — intentional.
**Completion criteria:** curl suite from §3.4 returns exactly the documented codes/bodies incl. 400 matrix (each bad-field case), 404 unknown ids, 503 flood.

### Phase 9: Frontend
**Goal:** Polished single-page UX matching the promise (drop → convert → compare → download).
**Prerequisites:** Phase 8.
**Tasks:**
1. index.html sections: header w/ GPU badge (fetch /healthz), dropzone (click+drag+paste-from-clipboard support), original thumbnail w/ dims, ratio segmented control (default 9:16), tier radio cards showing “~15 s” vs “~90 s” hints, align select (auto default), Convert button w/ disabled-until-valid logic, progress bar bound to stage labels (Detecting…/Placing…/Filling…/Polishing…), result panel = draggable-divider before/after slider (CSS clip-path technique, keyboard accessible), Download PNG button, subtle error toast surfacing API messages verbatim.
2. Polling loop 500 ms with exponential backoff after 30 s (cap 2 s), AbortController cancel button while queued.
3. History rail (localStorage of last 10 job_ids; click reloads slider from /api endpoints; entries whose files were purged show graceful placeholder).
4. Accessibility+dark theme, no external CDNs (fully offline page).
**Completion criteria:** manual pass: drag jpg → convert fast → slider scrubs smoothly → download works; clipboard-paste path works in Chrome+Edge; zero console errors.

### Phase 10: Verification, tuning, handoff docs
**Goal:** Provable correctness + tuned defaults + reproducible ops.
**Prerequisites:** All above.
**Tasks:**
1. `tests/test_pixel_guarantee.py`: parametrized over {ratios × tiers × 3 samples}: load upload + result, recompute placement paste_box via same pure function, assert `np.array_equal(result[y:y+h, x:x+w], original_scaled_region)` EXACTLY; additionally assert 2-px rim around paste differs from gray-fill color (proves fill happened everywhere).
2. Performance tuning session on this laptop: sweep FEATHER_PX {24,40,64} × OVERLAP_PX {32,48,80} on the 3 samples; pick defaults minimizing visible seam (SSIM across seam window) — bake winners into config.py with comment trail.
3. README.md: quickstart (the 4 commands), troubleshooting table (driver/OOM/artifact-missing symptoms→fixes), performance expectations table, how to swap ratios/add presets.
4. Final acceptance checklist run recorded in README (all §8 items ticked with dates/timings).
**Completion criteria:** pytest suite green incl. guarantee test across full matrix; documented chosen feather/overlap; a stranger can go clone→working in <15 min following README on this exact machine profile.

## 7. Error Handling Strategy

- **Boundary validation first** (route layer): type/enum/size/magic-bytes — reject cheaply with 400 before any GPU work.
- **Typed pipeline errors**: `QualityOOMError`, `ArtifactMissingError(name)`, `DecodeError` → mapped to specific user messages; anything unexpected → generic internal + logged traceback with job_id correlation.
- **VRAM policy**: prevention (sequential ModelSlot, fp16, low-res generation, tiling) > cure; cure ladder = single CPU/downscale retry for LaMa, hard-fail-with-guidance for PowerPaint.
- **Worker isolation**: pipeline crash can never take down uvicorn (dedicated thread, broad except, queue stays healthy); poisoned job skipped not retried (deterministic failures shouldn’t burn GPU minutes).
- **User-facing surfaces**: progress bar freezes never happen — every stage emits heartbeat pct; failures render actionable toast (“Run scripts\verify_assets.ps1” style pointers), never stack traces.
- **Data hygiene**: uploads deleted on terminal state + DELETE endpoint; startup purge bounds disk growth (24 h / 100 results); atomic writes (tmp+rename) everywhere files land.

## 8. Testing Checkpoints

- After Phase 1: healthz green on GPU; asset audit all-PASS (blocks everything else).
- After Phase 2: pytest geometry green; overlay demo eyeballed for 3 categories.
- After Phase 4 (first end-to-end milestone): FAST conversion of real news screenshot ≤25 s; row-std check proves no blank bands; manual zoom on ticker text = pixel perfect.
- After Phase 5: upscaler bench within time/VRAM budget.
- After Phase 6: QUALITY ≤95 s, VRAM ≤3.8 GB, seams acceptable at 200% zoom.
- After Phase 7: restart-recovery scenario passes; flood-control 503 verified.
- After Phase 8: full curl contract matrix matches spec.
- After Phase 9: full browser flow on Edge + Chrome; zero console errors.
- After Phase 10 (release gate): **pixel-guarantee test across 18-case matrix green**; README quickstart executed fresh by following only the doc.

## 9. Deployment Guide

This product deploys to the user’s laptop only:
1. `git clone` / open folder → `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1`
2. `scripts\fetch_local_weights.ps1`
3. First-time Quality tier only: `scripts\download_powerpaint.ps1` (~4–5 GB, resumable)
4. `scripts\verify_assets.ps1` → must be all green
5. Double-click `run.bat` (or `uvicorn app.main:app --host 127.0.0.1 --port 8000`) → browser opens automatically
6. Stop: Ctrl+C in console. Update flow: `git pull` + rerun steps 2/4 (weights gitignored, survive updates).
Optional autostart: shortcut to run.bat in shell:startup. No service/installer required by scope.

## 10. Agent Rules

1. **The pixel guarantee is law.** No stage may ever write inside the original-frame interior (minus the engineered overlap/feather ring). If any optimization tempts you to blur, rescale, or color-match the original region — stop and redesign.
2. One diffusion/inpaint model resident at ALL times. Every model access goes through `ModelSlot`. Never parallelize fill stages. Always `empty_cache()` on release.
3. Import `config.config` before torch in every entrypoint; never read env vars ad hoc in feature code.
4. No stubs, no TODOs, no placeholder engines. A tier/ratio/endpoint either fully works or the option doesn’t ship in the UI.
5. All user-visible errors are actionable sentences; raw tracebacks only in `logs\app.log`.
6. Respect FORCE_CPU in every model wrapper; code must not assume CUDA presence beyond setup-time warning.
7. File writes atomic (tmp + os.replace); every runtime dir created defensively on boot.
8. New dependencies require updating requirements.txt AND verify_assets/setup scripts in the same change.
9. Sample images in tests\samples are user-supplied real photos; never commit personal photos to git — samples dir is gitignored with a README explaining expected filenames.
10. When a fact about an external repo/API is uncertain (upstream PowerPaint file layout drift), verify against the checked-out copy in third_party at build time rather than assuming; record deviations in README troubleshooting.
