# Copies already-on-disk weights from sibling projects. Zero network usage.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot | Split-Path

$yoloSrc = "C:\Users\cools\My project\pngtopsd\backend\models\yolov8n-seg.pt"
$esrSrc  = "C:\Users\cools\My project\tv today  2\models\realesr-general-x4v3.pth"

foreach ($pair in @(@($yoloSrc, "$root\models\yolov8n-seg.pt"), @($esrSrc, "$root\models\realesr-general-x4v3.pth"))) {
    $src, $dst = $pair
    if (-not (Test-Path $src)) { Write-Host "MISSING SOURCE: $src" -ForegroundColor Red; exit 1 }
    Copy-Item -LiteralPath $src -Destination $dst -Force
    Write-Host ("copied {0} ({1:N1} MB)" -f (Split-Path $dst -Leaf), ((Get-Item $dst).Length / 1MB))
}
Write-Host "done." -ForegroundColor Green
