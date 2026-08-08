#!/usr/bin/env bash
# Fixes a real environment defect found while running this repo on an
# aarch64 (ARM64) host: wntr 1.5.0's bundled EPANET toolkit only ships an
# x86_64 shared library for Linux (wntr/epanet/toolkit.py hardcodes
# "libepanet/linux-x64/libepanet22.so" with no Linux-ARM branch at all,
# unlike its darwin-arm/darwin-x64 split), and wntr's own PyPI wheels have
# never published a manylinux aarch64 build either. Every live WNTR/EPANET
# simulation (scenario generation, plan/consequence verification, Stage E's
# oracle policy, etc.) fails closed with
# "OSError: ... libepanet22.so: cannot open shared object file" on such a
# host -- not a code bug in this repo, a missing native dependency.
#
# This script builds the OWA EPANET 2.2 C toolkit from source for the
# current architecture and drops it in at the exact path wntr expects,
# after backing up whatever was there -- the same governed EPANET 2.2
# toolkit API, just compiled for this host's real architecture instead of
# a foreign one that cannot even load. This makes live WNTR simulation
# work at all (verified: a real EpanetSimulator.run_sim() succeeds).
#
# IMPORTANT caveat, found real and confirmed 2026-08-08 (see
# scripts/restore_cycle_b2_train_scenario_cache.py's own docstring for the
# full account): this does NOT guarantee bit-for-bit reproduction of
# scenarios originally generated on a different CPU architecture. A
# same-seed same-config regeneration reproduced a simple (CLEAN-stage)
# scenario's artifact_sha256 exactly, but did NOT reproduce a more complex
# (ADVERSARIAL-stage, model-mismatch-perturbed) one -- real cross-
# architecture floating-point divergence in EPANET's iterative solver, not
# a config/determinism bug. Live simulation of NEW scenarios (this
# session, this host) is fine; treat any attempt to bit-for-bit reproduce
# a PRE-EXISTING x86_64-generated corpus's raw scenario arrays on this
# architecture as unverified until each individual scenario's own
# artifact_sha256 is checked, not assumed from one spot-check.
#
# Only needed on Linux hosts where `python3 -c "import wntr"` + a real
# EpanetSimulator.run_sim() raises the OSError above. Safe to re-run.
set -euo pipefail

WNTR_LIBEPANET_DIR="$(python3 -c 'import wntr, os; print(os.path.join(os.path.dirname(wntr.__file__), "epanet", "libepanet", "linux-x64"))')"
TARGET="${WNTR_LIBEPANET_DIR}/libepanet22.so"
BACKUP="${TARGET}.orig-arch.bak"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

git clone --depth 1 --branch v2.2 https://github.com/OpenWaterAnalytics/EPANET.git "${BUILD_DIR}/epanet-src"
cmake -S "${BUILD_DIR}/epanet-src" -B "${BUILD_DIR}/epanet-src/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build "${BUILD_DIR}/epanet-src/build" --target epanet2 -j"$(nproc)"

if [ ! -f "${BACKUP}" ] && [ -f "${TARGET}" ]; then
  cp "${TARGET}" "${BACKUP}"
fi
cp "${BUILD_DIR}/epanet-src/build/lib/libepanet2.so" "${TARGET}"

python3 -c "
import wntr
wn = wntr.network.WaterNetworkModel()
wn.add_reservoir('R1', base_head=100)
wn.add_junction('J1', base_demand=0.01, elevation=10)
wn.add_pipe('P1', 'R1', 'J1', length=1000, diameter=0.3)
wntr.sim.EpanetSimulator(wn).run_sim()
print('native EPANET simulation OK:', '${TARGET}')
"
