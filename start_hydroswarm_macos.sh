#!/usr/bin/env bash
# Native macOS launcher for HydroSwarm. Always uses the project-local
# .venv interpreter explicitly -- never an ambient/system Python -- and
# fails immediately if that venv does not exist rather than silently
# falling back to whatever `python` happens to resolve to on PATH.
set -euo pipefail

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
HOST="${HYDROSWARM_HOST:-127.0.0.1}"
PORT="${HYDROSWARM_PORT:-8765}"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "error: $VENV_PYTHON not found. Run ./setup_hydroswarm_macos.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Running readiness check ..."
if ! "$VENV_PYTHON" -m hydroswarm.cli self-test --human; then
  echo "" >&2
  echo "error: readiness self-test failed. Run ./setup_hydroswarm_macos.sh to diagnose and fix." >&2
  exit 1
fi

echo ""
echo "Starting HydroSwarm at http://${HOST}:${PORT}"
exec "$VENV_PYTHON" -m hydroswarm.cli start --host "$HOST" --port "$PORT"
