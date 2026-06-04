# Full local flow (no Docker): feed events then open dashboard instructions
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$py = Join-Path $Root ".venv\Scripts\python.exe"
$events = Join-Path $Root "output\events.jsonl"

if (-not (Test-Path $events)) {
    Write-Host "No events.jsonl — run pipeline first:"
    Write-Host "  python -m pipeline.run"
    exit 1
}

Write-Host "1) Start API in another terminal:"
Write-Host "     .\scripts\start_api.ps1"
Write-Host ""
Write-Host "2) Press Enter here after API is running..."
Read-Host

& $py scripts/feed_events.py --file output/events.jsonl
Write-Host ""
Write-Host "3) Test metrics:"
& curl.exe -s "http://127.0.0.1:8000/stores/STORE_BLR_002/metrics"
Write-Host ""
Write-Host "4) Optional live dashboard:"
Write-Host "     python scripts/live_dashboard.py"
