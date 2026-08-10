# Native Windows launcher for HydroSwarm. Always uses the project-local
# .venv interpreter explicitly -- never an ambient/system Python -- and
# fails immediately if that venv does not exist rather than silently
# falling back to whatever `python` happens to resolve to on PATH.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$hydroswarmHost = if ($env:HYDROSWARM_HOST) { $env:HYDROSWARM_HOST } else { "127.0.0.1" }
$Port = if ($env:HYDROSWARM_PORT) { $env:HYDROSWARM_PORT } else { "8765" }

if (-not (Test-Path $VenvPython)) {
    Write-Host "error: $VenvPython not found. Run .\setup_hydroswarm_windows.ps1 first." -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"

Write-Host "Running readiness check ..."
& $VenvPython -m hydroswarm.cli self-test --human
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "error: readiness self-test failed. Run .\setup_hydroswarm_windows.ps1 to diagnose and fix." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting HydroSwarm at http://${hydroswarmHost}:${Port}"
& $VenvPython -m hydroswarm.cli start --host $hydroswarmHost --port $Port
exit $LASTEXITCODE
