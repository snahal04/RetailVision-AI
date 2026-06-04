# Build and run the API with Docker Compose
$DockerBin = "C:\Program Files\Docker\Docker\resources\bin"
$Docker = Join-Path $DockerBin "docker.exe"
if (-not (Test-Path $Docker)) {
    $Docker = "docker"
    $DockerBin = $null
} elseif ($DockerBin -notin ($env:PATH -split ';')) {
    # docker-credential-desktop must be on PATH for image pulls
    $env:PATH = "$DockerBin;$env:PATH"
}

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "Checking Docker daemon..."
& $Docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Docker Desktop is not running."
    Write-Host "  1. Open 'Docker Desktop' from the Start menu"
    Write-Host "  2. Wait until it says 'Docker Desktop is running'"
    Write-Host "  3. Run this script again: .\scripts\docker_up.ps1"
    exit 1
}

Write-Host "Building and starting API on http://localhost:8000 ..."
& $Docker compose up --build -d
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Ready. Next:"
    Write-Host "  python scripts/feed_events.py --file output/events.jsonl"
    Write-Host "  curl http://localhost:8000/health"
    Write-Host "  docker compose logs -f api"
} else {
    Write-Host ""
    Write-Host "Build failed. If you see 'docker-credential-desktop' errors, run:"
    Write-Host "  .\scripts\fix_docker_credentials.ps1"
    exit 1
}
