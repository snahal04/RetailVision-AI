# Start Intelligence API without Docker (Windows)
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $Root

$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPython)) {
    Write-Error 'Missing .venv — run: python -m venv .venv; pip install -r requirements.txt'
    exit 1
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root 'data') | Out-Null
$env:DATABASE_URL = 'sqlite:///' + $Root.Replace('\', '/') + '/data/store_intelligence.db'
$env:PYTHONPATH = $Root

Write-Host 'Starting API at http://127.0.0.1:8000'
Write-Host 'Docs: http://127.0.0.1:8000/docs'
Write-Host 'Press Ctrl+C to stop'
& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
