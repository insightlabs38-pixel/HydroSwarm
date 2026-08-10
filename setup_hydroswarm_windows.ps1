# Native Windows setup for HydroSwarm: creates a project-local virtual
# environment, installs the CPU runtime, verifies the frozen HydroCore-v4
# release bundle, builds the frontend if needed, and runs the readiness
# self-test. Safe to re-run -- every step is idempotent.
#
# This script never installs anything outside .\.venv and does not run the
# full real-simulator test suite. Docker Desktop (WSL2) is the recommended
# production-equivalent path for exact-simulation latency -- see the note
# printed at the end.
#PowerShell strict mode -- fail fast and loud rather than continuing on an error.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequiredNodeMajor = 22

function Write-Info($Message) { Write-Host $Message }
function Fail($Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

Write-Info "HydroSwarm setup (Windows, $env:PROCESSOR_ARCHITECTURE)"
Write-Info ""

# --- 1. Locate a usable bootstrap Python (>=3.12, 64-bit) ------------------
$BootstrapPython = $null
foreach ($candidate in @("py -3.12", "python", "python3")) {
    $exe, $prefixArgs = $candidate.Split(" ", 2)
    $exeCmd = Get-Command $exe -ErrorAction SilentlyContinue
    if ($exeCmd) {
        try {
            $argList = @()
            if ($prefixArgs) { $argList += $prefixArgs }
            $argList += @("$ProjectRoot\scripts\setup_common.py", "check-python")
            & $exe @argList *> $null
            if ($LASTEXITCODE -eq 0) {
                $BootstrapPython = $candidate
                break
            }
        } catch {
            continue
        }
    }
}
if (-not $BootstrapPython) {
    Write-Host "error: no 64-bit Python 3.12+ interpreter found on PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "Install it from https://www.python.org/downloads/windows/"
    Write-Host "(check 'Add python.exe to PATH' during install), then re-run this script."
    exit 1
}
Write-Info "Bootstrap interpreter resolved via: $BootstrapPython"

# --- 2. Create the project-local virtual environment ------------------------
if (-not (Test-Path $VenvPython)) {
    Write-Info "Creating .venv ..."
    $exe, $prefixArgs = $BootstrapPython.Split(" ", 2)
    $argList = @()
    if ($prefixArgs) { $argList += $prefixArgs }
    $argList += @("-m", "venv", $VenvDir)
    & $exe @argList
} else {
    Write-Info "✓ .venv already exists"
}

# --- 3. Install the CPU runtime into .venv only ------------------------------
Write-Info "Installing dependencies into .venv (CPU wheels; never global) ..."
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" --quiet
& $VenvPython -m pip install -e ".[dev]" --quiet
Write-Info "✓ Runtime dependencies installed"

# --- 4. Verify the frozen V4 release bundle ---------------------------------
Write-Info "Verifying frozen HydroCore-v4 release bundle ..."
& $VenvPython "$ProjectRoot\scripts\setup_common.py" verify-bundle
if ($LASTEXITCODE -ne 0) {
    Fail "frozen HydroCore-v4 bundle failed verification (see above). This is a P0 blocker -- the release bundle under models\hydrocore-v4-release\ must be present and hash-verified before the app can serve the frozen architecture."
}
Write-Info "✓ Frozen HydroCore-v4 bundle verified"

# --- 5. Frontend: use prebuilt dist\ if present, else build with Node 22 ----
$FrontendStatus = & $VenvPython "$ProjectRoot\scripts\setup_common.py" frontend-status
if ($FrontendStatus -match '"built":\s*true') {
    Write-Info "✓ Prebuilt frontend found (frontend\dist) -- skipping frontend build"
} else {
    Write-Info "No prebuilt frontend found -- building from source."
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not $node) {
        Write-Host "error: Node.js 22+ is required to build the frontend and was not found on PATH." -ForegroundColor Red
        Write-Host ""
        Write-Host "Install it from https://nodejs.org/, then re-run this script."
        Write-Host "Alternatively, place a prebuilt frontend\dist\ directory to skip this step."
        exit 1
    }
    $nodeMajor = [int]((node -p "process.versions.node.split('.')[0]").Trim())
    if ($nodeMajor -lt $RequiredNodeMajor) {
        Fail "Node.js $RequiredNodeMajor+ required, found $(node --version). Install a newer Node and re-run."
    }
    Write-Info "Building frontend with Node $(node --version) ..."
    Push-Location "$ProjectRoot\frontend"
    try {
        npm.cmd ci
        npm.cmd run build
    } finally {
        Pop-Location
    }
    Write-Info "✓ Frontend built"
}

# --- 6. Run the readiness self-test -----------------------------------------
Write-Info ""
Write-Info "Running readiness self-test ..."
$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
& $VenvPython "$ProjectRoot\scripts\setup_common.py" self-test
$selfTestExit = $LASTEXITCODE
Write-Info ""
if ($selfTestExit -ne 0) {
    Fail "readiness self-test failed (see checklist above). Fix the flagged item and re-run this script."
}

Write-Info "Setup complete. Launch with:"
Write-Info "  .\start_hydroswarm_windows.ps1"
Write-Info ""
Write-Info "Note: HydroSwarm's optimized, production-equivalent runtime target is Linux"
Write-Info "(Docker Desktop with the WSL2 backend, amd64/arm64). Native Windows is a fully"
Write-Info "supported, correct install path but has materially higher latency for exact"
Write-Info "hydraulic/water-quality simulation -- see docs/INSTALLATION.md for why. This"
Write-Info "setup script does not run the full real-simulator test suite; use Docker for"
Write-Info "production-equivalent exact-simulation latency and full test coverage."
