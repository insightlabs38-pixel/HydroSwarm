#!/usr/bin/env sh
# Compatibility wrapper. Prefer the explicit platform launcher, which uses
# the project-local .venv interpreter and never an ambient system Python:
#   ./start_hydroswarm_linux.sh
#   ./start_hydroswarm_macos.sh
# This wrapper auto-detects the OS and delegates so existing muscle memory
# (`./start_hydroswarm.sh`) keeps working.
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$(uname -s)" in
  Darwin) exec "$PROJECT_ROOT/start_hydroswarm_macos.sh" "$@" ;;
  *)      exec "$PROJECT_ROOT/start_hydroswarm_linux.sh" "$@" ;;
esac
