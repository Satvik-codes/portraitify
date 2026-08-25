# Portraitify v3 "Layer-Aware Recomposition" — Implementation Plan

> SOURCE OF TRUTH for the v3 build. Re-read before starting any phase.
> Goal: `before.jpeg` → output matching the user's `after.png` reference in ONE click (Auto engine, preset selection only).

## 0. Locked decisions
- Default ratio **9:16** (1080×1920); add preset **2:3** (1024×1536).
- Smart mode fill engine preference: **SD1.5-inpaint + LCM-LoRA** (pending probe: fp16 vs fp32), fallback SD-Turbo fp32 → LaMa.
- **Background removal** for anchors AND logos (SAM2 alpha); red headline banner stays rectangular.
- One-click UX: Auto engine default, tier selector collapsed under Advanced.
- All models local; zero downloads. GPU fp16 compute is faulty → fp32 default everywhere unless probe proves a specific UNet clean.

## 1. Model roster (all on disk, paths verified)
| Role | Model | Path |
|---|---|---|
| Person/graphic boxes | YOLOv8n-seg | `models/yolov8n-seg.pt` (project) |
| Open-vocab graphic detection | YOLOWorld v2 + CLIP | `pngtopsd/backend/models/yolov8s-worldv2.pt` + `pngtopsd/backend/weights/clip/ViT-B-32.pt` |
| Cutout alpha (anchor, logos) | SAM 2.1 hiera base+ | HF cache `models--facebook--sam2.1-hiera-base-plus` |
| Photo cleanup fill | MI-GAN | torch hub `migan_traced.pt` |
| Cleanup fallback | Big-LaMa | torch hub `big-lama.pt` |
| **Band fill (primary)** | **SD1.5-inpainting + LCM-LoRA** | HF `models--botp--stable-diffusion-v1-5-inpainting` + `models--latent-consistency--lcm-lora-sdv1-5` |
| Fill fallback | SD-Turbo fp32 | HF `models--stabilityai--sd-turbo` |
| Final upscale (opt) | RealESRGAN x4v3 | `models/realesr-general-x4v3.pth` |

## 2. Target timing (per image, news graphic)
- Expected: **60–90 s** (LCM fp32 fill 40–60 s)
- Best: **20–35 s** (if LCM fp16 clean)
- Escalation ceiling: ~3 min (SD-Turbo), plain photos 10–20 s (LaMa)

## 3. Architecture
```
run_smart(tier)
 ├─ layers.detect_layers(img) ── no graphics? → continuation chain (fast→sdturbo→relayout)
 └─ SMART: SAM2 cutout alphas → MI-GAN clean photo holes → decide_paste(photo)
    → fill exposed canvas (LCM-inpaint → void gate → SD-Turbo → gate → LaMa)
    → compose photo byte-exact → paste cards (byte-exact, z-order, shadow)
    → fidelity asserts + voidcheck(exclude card boxes) → save
```
Layer kinds: `banner` (red headline, rect card), `strip` (खबरदार letters-only alpha), `corner_logo` (Aaj Tak badge alpha), `cutout` (anchor person alpha). Z-order: photo < cutout < corner_logo < strip < banner.

## 4. Detection spec (deterministic; before.jpeg = tuning reference)
- Banner: HSV S>140 V>90 → close 15 → comps; area ≥1.5 %, w≥15 %W, h≥3 %H.
- Strip: light rect (V>200 S<60) w≥20 %W h≥2.5 %H with internal Canny density ≥0.08 → SAM2 alpha (letters only).
- Corner logo: sat blob S>120, area 0.15–3 %, centroid within 18 % of corner → SAM2 alpha.
- Cutout person: YOLO conf≥0.5 + ring-uniformity std<14 → SAM2 box-prompt mask alpha.
- YOLOWorld+CLIP assist (phase 1.5, optional): prompts ["banner","logo","text"] as extra candidates validated by heuristics.

## 5. Phases
1. **P1 detection**: `pipeline/layers.py` + `tests/test_layers.py` (asserts on before.jpeg).
2. **P2 cleanup+planner**: MI-GAN hole inpaint; `plan_composition()` (rules: banner bottom w92 % m36; strip above w70 % gap24; corner logo top-right m24 rel-scaled; cutout left mapped rel; z-order; collision nudge).
3. **P3 smart engine**: SAM2 alphas; fill chain; `cards.py` paste w/ shadow; fidelity asserts (card == scaled crop bytes); voidcheck `exclude_boxes`; `run(tier="smart")` + run_smart routing (auto prefers smart when graphics found).
4. **P4 surface**: 2:3 preset; UI one-click (Auto default card, Advanced collapsed w/ tiers); engine badge shows chain.
5. **P5 tests**: fidelity, guarantee (card-aware), smoke e2e before.jpeg + newsshot.png; full regression.
6. **P6 docs+ship**: README v3, commits (logical splits), push, restart instructions.

## 6. Guarantees (law)
1. Card pixels = deterministic scaled source crop, byte-exact (asserted every smart job).
2. Photo region byte-exact except card footprints + fill mask.
3. Void gate on every output; card footprints excluded from stats, never weaken thresholds.
4. fp32 diffusion default; fp16 only behind probe pass / FORCE_FP16.
5. One model resident (ModelSlot); empty_cache on release.
6. No stubs; thresholds in config; deterministic detection.

## 7. Probe (run first)
SD1.5-inpaint+LCM on newsshot-composite: fp16 → black fraction + s/step; if NaN → fp32 timing. Result picks fill dtype + expected timing; record here: ____________
