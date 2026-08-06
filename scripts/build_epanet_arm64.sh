#!/usr/bin/env bash
# Build EPANET 2.2's shared library for linux-arm64 and install it into the
# active Python environment's wntr package.
#
# wntr 1.5.0 (the version this repo pins, `wntr>=1.3`) bundles prebuilt EPANET
# toolkit binaries for windows-x64, darwin-x64, darwin-arm, and linux-x64 only
# (see wntr/epanet/toolkit.py: the Linux branch is a bare `else` with no
# architecture check, unlike the darwin branch). There is no linux-arm64
# build anywhere upstream (no wheel, no sdist source, no apt package). Every
# WNTR/EPANET call -- corpus generation, the deterministic_replay and
# mask_round_trip corpus gates, Strategist verification, live inference --
# fails on an aarch64 host with:
#   OSError: .../libepanet/linux-x64/libepanet22.so: cannot open shared object file
# (a wrong-ELF-class load failure, not a missing file -- the x86-64 binary is
# physically present, ctypes/dlopen just cannot execute it).
#
# Fix: EPANET's toolkit (github.com/OpenWaterAnalytics/EPANET, tag v2.2 --
# this is the actual 2.2.0 release; the newer github.com/USEPA/EPANET mirror
# only tags v2.3.0-beta) is open source, BSD-licensed, and CMake-based.
# ctypes.cdll.LoadLibrary does not care what its containing directory is
# *named* -- only that the .so matches the host architecture -- so building
# EPANET 2.2 for aarch64 and dropping it at wntr's hardcoded
# "libepanet/linux-x64/libepanet22.so" path works without touching wntr or
# this repo's Python code at all.
#
# Idempotent and safe to re-run after every `uv sync` (a fresh sync
# reinstalls wntr's wheel, which reverts this patch).
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
  echo "build_epanet_arm64.sh: host arch is '$ARCH', not aarch64/arm64 -- nothing to do." >&2
  exit 0
fi

PY=${PYTHON:-uv run python3}
WNTR_EPANET_DIR=$($PY -c "import wntr, os; print(os.path.join(os.path.dirname(wntr.__file__), 'epanet', 'libepanet', 'linux-x64'))")
TARGET="$WNTR_EPANET_DIR/libepanet22.so"

if [ -f "$TARGET" ] && command -v readelf >/dev/null 2>&1; then
  MACHINE=$(readelf -h "$TARGET" 2>/dev/null | awk -F: '/Machine/{print $2}' | tr -d ' ')
  if [ "$MACHINE" = "AArch64" ]; then
    echo "build_epanet_arm64.sh: $TARGET is already AArch64 -- nothing to do."
    exit 0
  fi
fi

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

echo "build_epanet_arm64.sh: cloning OpenWaterAnalytics/EPANET @ v2.2 ..."
git clone --quiet --depth 1 --branch v2.2 \
  https://github.com/OpenWaterAnalytics/EPANET.git "$WORK_DIR/epanet-src"

echo "build_epanet_arm64.sh: building libepanet2 (Release, tests off) ..."
cmake -S "$WORK_DIR/epanet-src" -B "$WORK_DIR/epanet-src/build" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF >/dev/null
cmake --build "$WORK_DIR/epanet-src/build" --target epanet2 --config Release \
  -- -j"$(nproc)" >/dev/null

BUILT="$WORK_DIR/epanet-src/build/lib/libepanet2.so"
[ -f "$BUILT" ] || { echo "build_epanet_arm64.sh: build did not produce $BUILT" >&2; exit 1; }

mkdir -p "$WNTR_EPANET_DIR"
cp "$BUILT" "$TARGET"
echo "build_epanet_arm64.sh: installed aarch64 libepanet22.so -> $TARGET"

$PY -c "
import wntr
wn = wntr.network.WaterNetworkModel('data/topologies/loop-grid.inp')
results = wntr.sim.EpanetSimulator(wn).run_sim()
assert results.node['pressure'].shape[1] > 0
print('build_epanet_arm64.sh: verified -- EpanetSimulator ran a real simulation OK')
"
