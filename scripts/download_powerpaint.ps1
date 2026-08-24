# One-time download of Quality-tier artifacts (~4-5 GB total). RUN ONLY ON USER COMMAND.
param([switch]$WhatIfOnly)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path
$tp = Join-Path $root "third_party"
$ppDir = Join-Path $tp "PowerPaint_v2"
$sdDir = Join-Path $tp "sd15_base"

Write-Host "== Plan =="
Write-Host "1. git clone https://huggingface.co/Sanster/PowerPaint_v2 -> $ppDir"
Write-Host "   (module code + BrushNet fp16 weights + task tokenizer)"
Write-Host "2. huggingface_hub.snapshot_download stable-diffusion-v1-5/stable-diffusion-v1-5 (fp16 subset) -> $sdDir"
Write-Host "   (unet fp16 safetensors + vae fp16 + text_encoder + tokenizer + scheduler)"
if ($WhatIfOnly) { Write-Host "WhatIfOnly set - nothing downloaded."; exit 0 }

if (-not (Test-Path $ppDir)) {
    git lfs install 2>$null | Out-Null
    git clone --depth 1 https://huggingface.co/Sanster/PowerPaint_v2 $ppDir
    if ($LASTEXITCODE -ne 0) { Write-Host "git clone failed - is git installed?" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "PowerPaint_v2 dir exists - skipping clone."
}

$py = @"
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir=r"$sdDir",
    allow_patterns=[
        "*.json",
        "text_encoder/model.safetensors", "text_encoder/config.json",
        "tokenizer/**",
        "unet/diffusion_pytorch_model.fp16.safetensors", "unet/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
        "scheduler/scheduler_config.json",
    ],
)
print("SD1.5 subset at:", p)
"@
$tmp = Join-Path $env:TEMP "h2v_sd15_dl.py"
$py | Set-Content $tmp -Encoding UTF8
python $tmp
if ($LASTEXITCODE -ne 0) { Write-Host "snapshot_download failed" -ForegroundColor Red; exit 1 }

Write-Host "Quality-tier artifacts ready." -ForegroundColor Green
Write-Host "Next: run scripts\verify_assets.ps1 to confirm."
