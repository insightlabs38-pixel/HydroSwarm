#!/usr/bin/env sh
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m hydroswarm.cli self-test
exec python -m hydroswarm.cli start --host 127.0.0.1 --port 8765
