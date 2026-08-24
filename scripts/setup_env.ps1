# Environment verification. System Python path assumed (C:\Python312).
# Optional: -Venv to create an isolated venv instead of using system packages.
param([switch]$Venv)
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot | Split-Path

if ($Venv) {
    py -3 -m venv "$root\.venv"
    & "$root\.venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    & "$root\.venv\Scripts\python.exe" -m pip install -r "$root\requirements.txt"
    Write-Host "venv created at .venv (activate before run.bat)" -ForegroundColor Green
    exit $LASTEXITCODE
}

Write-Host "== python ==" 
python --version
$fail = 0
foreach ($pkg in @("torch", "ultralytics", "cv2", "PIL", "fastapi", "uvicorn",
                   "diffusers", "transformers", "realesrgan", "simple_lama_inpainting",
                   "multipart", "dotenv")) {
    python -c "import $pkg" 2>$null
    if ($?) { Write-Host ("PASS  {0}" -f $pkg) } else { Write-Host ("MISSING  {0}" -f $pkg) -ForegroundColor Red; $fail++ }
}
Write-Host "== gpu =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.version.cuda)"
if (-not $?) { $fail++ }

if ($fail -eq 0) { Write-Host "`nENVIRONMENT OK - next: scripts\fetch_local_weights.ps1" -ForegroundColor Green; exit 0 }
Write-Host "`n$fail item(s) missing - pip install them (see requirements.txt)" -ForegroundColor Yellow
exit 1
