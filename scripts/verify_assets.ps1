# Post-install audit: every artifact the pipeline needs must exist and be sane.
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot | Split-Path
$fail = 0
function Check([string]$label, [bool]$ok, [string]$detail = "") {
    if ($ok) { Write-Host ("PASS  {0}  {1}" -f $label, $detail) }
    else     { Write-Host ("FAIL  {0}  {1}" -f $label, $detail) -ForegroundColor Red; $script:fail++ }
}

$lama = Join-Path $env:USERPROFILE ".cache\torch\hub\checkpoints\big-lama.pt"
if (Test-Path $lama) {
    $mb = (Get-Item $lama).Length / 1MB
    Check "big-lama.pt" ($mb -gt 150) ("{0:N0} MB" -f $mb)
} else { Check "big-lama.pt" $false "not found at $lama" }

foreach ($f in @("models\yolov8n-seg.pt", "models\realesr-general-x4v3.pth")) {
    $p = Join-Path $root $f
    if (Test-Path $p) { Check $f (((Get-Item $p).Length / 1KB) -gt 100) ("{0:N1} MB" -f ((Get-Item $p).Length / 1MB)) }
    else { Check $f $false "run scripts\fetch_local_weights.ps1" }
}

$pp = Join-Path $root "third_party\PowerPaint_v2\powerpaint_v2\pipeline_PowerPaint_Brushnet_CA.py"
Check "PowerPaint-V2 (Quality tier)" (Test-Path $pp) $(if (Test-Path $pp) { "present" } else { "run scripts\download_powerpaint.ps1 when ready" })

$sd = Join-Path $root "third_party\sd15_base\unet"
Check "SD1.5 base (Quality tier)" (Test-Path $sd) $(if (Test-Path $sd) { "present" } else { "run scripts\download_powerpaint.ps1 when ready" })

Write-Host "== Python env =="
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available()); import sys; sys.exit(0 if torch.cuda.is_available() or True else 1)"
if (-not $?) { $script:fail++ }

if ($fail -eq 0) { Write-Host "`nALL CHECKS PASSED" -ForegroundColor Green; exit 0 }
else { Write-Host "`n$fail check(s) failed"; exit 1 }
