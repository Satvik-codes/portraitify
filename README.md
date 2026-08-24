# Portraitify

**Horizontal → vertical image converter that thinks.** Drop any landscape photo — news screenshots with tickers, portraits, landscapes — and it detects the subjects, smartly places the untouched original on a tall canvas, and AI-fills every remaining pixel so the result looks like it was shot vertical. No blank bars, no stretching.

Runs 100% locally on a modest GPU (built & tuned on a GTX 1650 4GB). Nothing ever leaves localhost.

## How it works

```
detect   → YOLOv8n-seg finds people/objects; saliency fallback
place    → subject-aware placement of the ORIGINAL frame on the target canvas
           (original pixels are never modified — text stays crisp)
fill     → FAST: Big-LaMa (~10-20s)      QUALITY: PowerPaint-V2 (~90s)
polish   → RealESRGAN upscale (Quality tier only)
compose  → byte-identical re-paste + seam feathering, guarantee asserted
```

The **pixel guarantee**: every output's original region is verified byte-for-byte against the input before saving. A violated guarantee aborts the job instead of shipping.

| | Fast | Quality |
|---|---|---|
| Engine | Big-LaMa | PowerPaint-V2 (BrushNet) |
| Time / image | ~10–20 s | ~75–95 s |
| Best for | skies, walls, studio backgrounds | semantic scenes needing new content |
| Weights needed | cached automatically | one-time ~4–5 GB download |

## Quickstart

```powershell
# 1. environment check + deps (system python 3.12 assumed)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. copy local weights from sibling projects (no downloads)
powershell -ExecutionPolicy Bypass -File scripts\fetch_local_weights.ps1

# 3. optional Quality tier (~4-5 GB, resumable)
powershell -ExecutionPolicy Bypass -File scripts\download_powerpaint.ps1

# 4. audit everything
powershell -ExecutionPolicy Bypass -File scripts\verify_assets.ps1

# 5. run
.\run.bat          # opens http://127.0.0.1:8000
```

Drag an image in, pick a ratio (9:16 / 4:5 / 1:1), Convert, slide the before/after handle, download.

## Tests

```powershell
python tests\test_placement_math.py        # geometry + mask invariants (16 checks)
python tests\smoke_e2e.py --tier fast      # end-to-end conversion + timings
python tests\test_pixel_guarantee.py       # byte-identity matrix (the big one)
python tests\test_pixel_guarantee.py --full
```

Drop real photos into `tests/samples/` (gitignored) to test on your own material.

## Tuning knobs (`config/config.py`)

| Knob | Default | Effect |
|---|---|---|
| `FEATHER_PX` | 40 | mask blur at seams — higher = softer blend |
| `OVERLAP_PX` | 48 | how far filling eats into the frame border for context |
| `RAMP_PX` | 16 | background-side softening width before the paste edge |
| `PP_GEN_LONG_SIDE` | 1120 | Quality-tier generation resolution (VRAM budget) |

Defaults were picked for news-screenshot-style material on a GTX 1650; sweep them on your own samples via `KEEP_DEBUG=true` to inspect masks and fills per job under `debug/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `verify_assets` FAIL on big-lama.pt | open IOPaint once or place `big-lama.pt` in `%USERPROFILE%\.cache\torch\hub\checkpoints\` |
| Quality button disabled in UI | run `scripts\download_powerpaint.ps1` (needs your go-ahead — it's the only download) |
| "GPU ran out of memory" toast | use Fast tier, smaller source, or close VRAM-hungry apps |
| server restarted mid-job | job auto-marked failed; just resubmit |
| CUDA unavailable in healthz | update NVIDIA driver; check `nvidia-smi` |

## Layout

```
app/         FastAPI server, routes, job engine, static UI
pipeline/    detect · placement · fillers(lama/powerpaint) · upscale · compose
config/      single source of truth for every knob & path
scripts/     env setup, weight fetching, downloads, audits
tests/       placement math · smoke e2e · pixel-guarantee matrix
models/      local weights (gitignored)   third_party/  PP-V2 + SD1.5 (gitignored)
```

See `implementation-plan.md` for the full architecture the build followed.
