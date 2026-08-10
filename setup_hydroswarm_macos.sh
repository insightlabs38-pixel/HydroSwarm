#!/usr/bin/env bash
# Native macOS setup for HydroSwarm: creates a project-local virtual
# environment, installs the CPU runtime, verifies WNTR/EPANET and the
# frozen HydroCore-v4 release bundle, builds the frontend if needed, and
# runs the readiness self-test. Safe to re-run -- every step is idempotent.
#
# This script never uses `sudo`, never installs anything outside ./.venv,
# and does not assume Rosetta on Apple Silicon. If a required tool is
# missing, it prints the exact command to install it and exits.
set -euo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
REQUIRED_NODE_MAJOR=22

log()  { printf '%s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }

ARCH=$(uname -m)
case "$ARCH" in
  arm64)  CHIP_LABEL="Apple Silicon (arm64)" ;;
  x86_64) CHIP_LABEL="Intel (x86_64)" ;;
  *)      CHIP_LABEL="$ARCH" ;;
esac
log "HydroSwarm setup (macOS, $CHIP_LABEL)"
log ""

if [ "$ARCH" = "x86_64" ] && [ "$(sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
  log "note: running under Rosetta 2 translation. A native arm64 Python is recommended on Apple Silicon for best performance; this script does not require or assume Rosetta."
fi

# --- 1. Locate a usable bootstrap Python (>=3.12, 64-bit) ------------------
BOOTSTRAP_PYTHON=""
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" "$PROJECT_ROOT/scripts/setup_common.py" check-python >/dev/null 2>&1; then
      BOOTSTRAP_PYTHON="$candidate"
      break
    fi
  fi
done
if [ -z "$BOOTSTRAP_PYTHON" ]; then
  cat >&2 <<'EOF'
error: no Python 3.12+ (64-bit) interpreter found on PATH.

Install it with Homebrew:
  brew install python@3.12

or download it from https://www.python.org/downloads/macos/
Then re-run this script.
EOF
  exit 1
fi
log "✓ Bootstrap interpreter: $(command -v "$BOOTSTRAP_PYTHON") ($("$BOOTSTRAP_PYTHON" --version 2>&1))"

# --- 2. Create the project-local virtual environment ------------------------
if [ ! -x "$VENV_PYTHON" ]; then
  log "Creating .venv ..."
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
else
  log "✓ .venv already exists"
fi

# --- 3. Install the CPU runtime into .venv only ------------------------------
log "Installing dependencies into .venv (CPU wheels; never global) ..."
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
"$VENV_PYTHON" -m pip install "torch>=2.5" --quiet
"$VENV_PYTHON" -m pip install -e ".[dev]" --quiet
log "✓ Runtime dependencies installed"

# --- 4. Verify WNTR/EPANET import + the frozen V4 release bundle -----------
log "Verifying WNTR/EPANET import ..."
if ! "$VENV_PYTHON" -c "import wntr; wntr.sim.EpanetSimulator" >/dev/null 2>&1; then
  fail "WNTR failed to import its EPANET simulator binding. Reinstall with: .venv/bin/python -m pip install --force-reinstall wntr"
fi
log "✓ WNTR/EPANET import verified"

log "Verifying frozen HydroCore-v4 release bundle ..."
if ! "$VENV_PYTHON" "$PROJECT_ROOT/scripts/setup_common.py" verify-bundle; then
  fail "frozen HydroCore-v4 bundle failed verification (see above). This is a P0 blocker -- the release bundle under models/hydrocore-v4-release/ must be present and hash-verified before the app can serve the frozen architecture."
fi
log "✓ Frozen HydroCore-v4 bundle verified"

# --- 5. Frontend: use prebuilt dist/ if present, else build with Node 22 ----
FRONTEND_STATUS=$("$VENV_PYTHON" "$PROJECT_ROOT/scripts/setup_common.py" frontend-status)
if echo "$FRONTEND_STATUS" | grep -q '"built": true'; then
  log "✓ Prebuilt frontend found (frontend/dist) -- skipping frontend build"
else
  log "No prebuilt frontend found -- building from source."
  if ! command -v node >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error: Node.js 22+ is required to build the frontend and was not found on PATH.

Install it with Homebrew:
  brew install node@22

or via nvm: `nvm install 22`. Then re-run this script. Alternatively, place
a prebuilt frontend/dist/ directory to skip this step.
EOF
    exit 1
  fi
  NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]')
  if [ "$NODE_MAJOR" -lt "$REQUIRED_NODE_MAJOR" ]; then
    fail "Node.js $REQUIRED_NODE_MAJOR+ required, found $(node --version). Install a newer Node (e.g. \`brew install node@22\`) and re-run."
  fi
  log "Building frontend with Node $(node --version) ..."
  (cd "$PROJECT_ROOT/frontend" && npm ci && npm run build)
  log "✓ Frontend built"
fi

# --- 6. Run the readiness self-test -----------------------------------------
log ""
log "Running readiness self-test ..."
if ! PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$VENV_PYTHON" "$PROJECT_ROOT/scripts/setup_common.py" self-test; then
  log ""
  fail "readiness self-test failed (see checklist above). Fix the flagged item and re-run this script."
fi
log ""

log "Setup complete. Launch with:"
log "  ./start_hydroswarm_macos.sh"
