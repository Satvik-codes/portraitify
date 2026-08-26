# Portraitify

**Horizontal → vertical image converter that thinks.** Drop any landscape photo — news screenshots with tickers, portraits, landscapes — and it detects the subjects, then fills every pixel of the vertical frame so nothing is ever blank: either the scene continues naturally, or the image's own elements are rearranged into a composed vertical card.

Runs 100% locally (built on a GTX 1650 4GB laptop). Nothing leaves localhost.

## The engine chain (why results are never black)

Every conversion runs through a **void gate** — bands are scored for uniform/featureless content after filling. If an engine's output fails, the next one takes over automatically. No selection can ever ship dead bands.

| Order | Engine | What it does | Time* |
|---|---|---|---|
| 1 | **Fast (Big-LaMa)** | mirror-seeded texture continuation of sky/walls/background | ~15 s |
| 2 | **SD-Turbo (fp32)** | semantic img2img scene continuation | ~90 s |
| 3 | **Re-layout** | the image's own elements (hero subject, banner, logo) recomposed as a vertical card over a blurred backdrop | ~20 s |
| opt-in | **Quality (PowerPaint-V2)** | full semantic outpainting — needs healthy fp16; auto-falls-back to the chain above when the GPU can't | ~4 min |

*Times measured on a GTX 1650 Mobile (4GB).

The chosen engine is reported per result (`engine_chain` in the job metadata / shown in the UI status line).

## Known GPU caveat

GTX 16-series cards with recent NVIDIA drivers (610.x) exhibit broken fp16 compute: NaN-corrupted diffusion output and inverted fp16/fp32 throughput. This app detects and routes around it (fp32 paths + escalation). If you roll back to a 55x.xx driver and `python scripts\gpu_probe.py` reports <40 ms matmul, add `FORCE_FP16=true` to `.env` to re-enable fast fp16 everywhere.

## Quickstart

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1            # env + GPU check
powershell -ExecutionPolicy Bypass -File scripts\fetch_local_weights.ps1  # copies local weights
powershell -ExecutionPolicy Bypass -File scripts\download_powerpaint.ps1  # optional Quality tier (~4-5 GB)
powershell -ExecutionPolicy Bypass -File scripts\verify_assets.ps1        # all-PASS gate
.\run.bat                                                                 # http://127.0.0.1:8000
```

Drop an image → pick ratio (9:16 / 4:5 / 1:1) → tier → Convert → compare-slider → download.

## The pixel guarantee

For continuation tiers, the original frame region in every output is verified **byte-for-byte identical** to the input before saving (`tests\test_pixel_guarantee.py`). Text, tickers and logos stay crisp by construction. Re-layout composes from exact source crops instead, so fidelity holds there too (`fidelity_ok` in job metadata).

## Tests

```powershell
python tests\test_placement_math.py        # geometry + mask invariants
python tests\smoke_e2e.py --tier fast      # end-to-end + timings
python tests\test_pixel_guarantee.py       # byte-identity matrix
```

## Tuning knobs (`config/config.py`)

| Knob | Default | Effect |
|---|---|---|
| `FEATHER_PX` / `OVERLAP_PX` | 40 / 48 | seam blend vs context depth for fillers |
| `PP_GEN_LONG_SIDE` / `PP_STEPS` | 576 / 8 | PowerPaint resolution/steps (VRAM-time budget) |
| `FORCE_FP16` | false | flip ON only on verified-healthy fp16 hardware |

## Troubleshooting

| Symptom | Fix |
|---|---|
| black/dark bands in Quality | expected on fp16-broken GPUs — use Auto; see GPU caveat above |
| `verify_assets` FAIL big-lama.pt | place `big-lama.pt` in `%USERPROFILE%\.cache\torch\hub\checkpoints\` |
| Quality button slow (~4 min) | fp32 offload path; Fast/Auto recommended on 4 GB cards |
| server restarted mid-job | job auto-marked failed; resubmit |

## Layout

```
app/         FastAPI server, routes, job engine, static UI
pipeline/    detect · placement · voidcheck · relayout · fillers(lama/sdturbo/powerpaint) · upscale · compose
config/      single source of truth for every knob & path
scripts/     env setup, weight fetching, downloads, audits, GPU probes
tests/       placement math · smoke e2e · pixel-guarantee matrix
```

See `implementation-plan.md` for the full architecture.
