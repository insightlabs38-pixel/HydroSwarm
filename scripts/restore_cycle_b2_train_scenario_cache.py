"""Restore data/learning-v2/cycle-b2/scenarios/train/*.npz on an environment
where they are missing (this repo's .gitignore deliberately excludes
data/learning-v2/**/scenarios/**/*.npz -- only the JSONL manifests and the
pre-built tensor shards are committed; the raw per-scenario simulation
arrays are treated as a regenerable cache, not a governed artifact).

**2026-08-08 finding, real and empirically confirmed -- read before relying
on this script's output for anything semantically load-bearing**:
same-seed same-config WNTR/EPANET regeneration is NOT guaranteed
bit-for-bit reproducible across CPU architectures. A real run of this
script on an aarch64 host reproduced golden-reference's CLEAN-stage
scenario (index 0, seed 71000) exactly, but its ADVERSARIAL-stage
scenario (index 4, seed 71400, with model_mismatch flags set) did NOT
match cycle-b2's own committed artifact_sha256 -- caught by this script's
own fail-closed verification below, not silently accepted. Root cause is
believed to be genuine cross-architecture floating-point divergence in
EPANET's iterative hydraulic/water-quality solver (more complex/longer
simulation paths appear more likely to diverge), not a bug in this
script's config reconstruction -- the mismatched scenario's scenario_id
matched exactly (proving the config was reconstructed correctly); only
the simulated array VALUES differed. Because of this, do not assume a
successful run of this script on any given host reproduces the FULL
train split correctly merely because some scenarios verify -- every
scenario must pass its own fail-closed check, and if even one fails (as
happened here), the correct response is to stop, not to skip and
continue. scripts/generate_ood_extension_corpus.py no longer depends on
this script for its signature-library needs for exactly this reason (it
now loads cycle-b2's own already-committed signatures/<family>.json
directly instead -- see that script's own docstring). This script remains
useful for narrow, verified spot-checks (e.g. confirming a specific
scenario reproduces on a given host) but should not be treated as a
general-purpose full-cache restoration tool until/unless a host is
independently confirmed to reproduce cycle-b2 bit-for-bit end to end.

This script does NOT regenerate cycle-b2 (restriction #3: cycle-b2 is
immutable and must never be regenerated to produce different values). It
replays generate_cycle_b_corpus.py's own unmodified
_generate_topology_scenarios() -- the exact original code path, same
seed_base formula, same per-scenario ScenarioGenerationConfig construction
-- and writes ONLY the raw .npz array files (never touching the already-
committed manifests/train.jsonl or any other split). Every regenerated
scenario is then verified for real, not assumed: load_generated_scenarios
(the same function generate_ood_extension_corpus.py itself calls) already
recomputes each scenario's artifact_sha256 from the loaded arrays and
raises ValueError on any mismatch against the committed manifest -- this
script calls that same loader immediately after writing, per family, and
fails closed if the committed corpus's recorded hashes do not match what
this environment's WNTR/numpy reproduces bit-for-bit. On failure, this
script does not clean up whatever it already wrote -- the caller is
responsible for removing any partial, unverified scenarios/train/
contents before treating the directory as trustworthy again.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
from generate_cycle_b_corpus import TRAIN_TOPOLOGIES, _generate_topology_scenarios  # noqa: E402

from hydroswarm.data.scenarios import DatasetSplit, load_generated_scenarios  # noqa: E402


def _write_npz_only(scenarios_root: Path, scenario) -> Path:  # noqa: ANN001
    split_root = scenarios_root / scenario.manifest.split.value
    split_root.mkdir(parents=True, exist_ok=True)
    artifact = split_root / f"{scenario.manifest.scenario_id}.npz"
    np.savez_compressed(
        artifact,
        timestamps_seconds=scenario.timestamps_seconds,
        truth_concentration=scenario.truth_concentration,
        observed_concentration=scenario.observed_concentration,
        observation_mask=scenario.observation_mask,
        frozen_mask=scenario.frozen_mask,
        communication_outage_mask=scenario.communication_outage_mask,
        timestamp_jitter_seconds=scenario.timestamp_jitter_seconds,
        flow_reversal_mask=scenario.flow_reversal_mask,
        drift_mask=scenario.drift_mask,
        unit_mismatch_mask=scenario.unit_mismatch_mask,
    )
    return artifact


def main() -> int:
    cycle_b2_root = Path("data/learning-v2/cycle-b2")
    scenarios_root = cycle_b2_root / "scenarios"
    started = time.perf_counter()

    for topology_index, (network_family, loader) in enumerate(TRAIN_TOPOLOGIES):
        family_started = time.perf_counter()
        network = loader()
        seed_base = 71_000 + topology_index * 10_000_000  # matches generate_cycle_b_corpus.py main()'s own default --seed
        by_split = _generate_topology_scenarios(network_family=network_family, network=network, seed_base=seed_base)
        train_scenarios = by_split[DatasetSplit.TRAIN]
        written = 0
        for scenario, _randomized_network in train_scenarios:
            _write_npz_only(scenarios_root, scenario)
            written += 1
        # Fail-closed verification: reload through the SAME loader
        # generate_ood_extension_corpus.py itself uses, which independently
        # recomputes artifact_sha256 from the arrays just written and
        # compares it against the already-committed manifests/train.jsonl.
        # A mismatch here means this environment's WNTR/numpy does not
        # reproduce cycle-b2 bit-for-bit -- a real problem to stop and
        # report, not paper over.
        reloaded = [
            s
            for s in load_generated_scenarios(scenarios_root, DatasetSplit.TRAIN)
            if s.manifest.network_family == network_family
        ]
        if len(reloaded) != written:
            raise ValueError(
                f"{network_family}: wrote {written} train .npz files but only {len(reloaded)} verified "
                "against the committed manifest -- restoration is incomplete or non-reproducible"
            )
        print(
            f"{network_family}: restored and verified {written} train .npz files "
            f"({time.perf_counter() - family_started:.1f}s)"
        )

    print(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
