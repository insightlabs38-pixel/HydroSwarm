#!/usr/bin/env bash
# Native Linux setup for HydroSwarm: creates a project-local virtual
# environment, installs the CPU runtime, verifies the frozen HydroCore-v4
# release bundle, builds the frontend if needed, and runs the readiness
# self-test. Safe to re-run -- every step is idempotent.
#
# This script never touches system package managers (apt/yum/pacman) and
# never installs anything outside ./.venv. If a required tool is missing,
# it prints the exact command to install it and exits, rather than guessing.
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
  x86_64)  PLATFORM_LABEL="linux/amd64" ;;
  aarch64|arm64) PLATFORM_LABEL="linux/arm64" ;;
  *) PLATFORM_LABEL="linux/$ARCH" ;;
esac
log "HydroSwarm setup (Linux, $PLATFORM_LABEL)"
log ""

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

Install it with your distribution's package manager, e.g.:
  Debian/Ubuntu:  sudo apt-get install python3.12 python3.12-venv
  Fedora:         sudo dnf install python3.12
  Arch:           sudo pacman -S python

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
"$VENV_PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" --quiet
"$VENV_PYTHON" -m pip install -e "." --quiet
log "✓ Runtime dependencies installed"

# --- 4. Verify the frozen V4 release bundle ---------------------------------
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

Install it (e.g. via https://nodejs.org/ or your distribution's package
manager, or nvm: `nvm install 22`), then re-run this script. Alternatively,
place a prebuilt frontend/dist/ directory to skip this step.
EOF
    exit 1
  fi
  NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]')
  if [ "$NODE_MAJOR" -lt "$REQUIRED_NODE_MAJOR" ]; then
    fail "Node.js $REQUIRED_NODE_MAJOR+ required, found $(node --version). Install a newer Node (e.g. \`nvm install 22\`) and re-run."
  fi
  log "Building frontend with Node $(node --version) ..."
  (cd "$PROJECT_ROOT/frontend" && npm ci && npm run build)
  log "✓ Frontend built"
fi

# --- 6. Run the readiness self-test -----------------------------------------
log ""
log "Running readiness self-test ..."
if ! PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$VENV_PYTHON" "$PROJECT_ROOT/scripts/setup_common.py" self-test --strict; then
  log ""
  fail "readiness self-test failed (see checklist above). Fix the flagged item and re-run this script."
fi
log ""

log "Setup complete. Launch with:"
log "  ./start_hydroswarm_linux.sh"
