$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $ProjectRoot "frontend")
try {
    npm.cmd ci
    npm.cmd run lint
    npm.cmd run test -- --run
    npm.cmd run build
} finally {
    Pop-Location
}
