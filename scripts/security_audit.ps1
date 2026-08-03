$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    uv export --frozen --no-dev --no-emit-project --format requirements-txt -o reports/results/requirements.lock.txt
    python -m pip_audit -r reports/results/requirements.lock.txt --require-hashes -f json -o reports/results/pip-audit.json
} finally {
    Pop-Location
}
