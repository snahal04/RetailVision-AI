# Fix: docker-credential-desktop not found in PATH
# Adds Docker Desktop bin to the current user's PATH permanently.

$DockerBin = "C:\Program Files\Docker\Docker\resources\bin"
if (-not (Test-Path $DockerBin)) {
    Write-Error "Docker Desktop bin folder not found. Is Docker Desktop installed?"
    exit 1
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -split ';' -contains $DockerBin) {
    Write-Host "Docker bin already on user PATH."
} else {
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$DockerBin", "User")
    Write-Host "Added to user PATH: $DockerBin"
    Write-Host "Restart PowerShell / Cursor terminal, then run .\scripts\docker_up.ps1"
}

$env:PATH = "$DockerBin;$env:PATH"
Write-Host "Current session PATH updated."
