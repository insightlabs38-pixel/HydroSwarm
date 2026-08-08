"""Versioned classical-signature generation policy (core-issues5.txt
Section 4, P0 blocker).

HydroCore consumes the classical prior as a model input (`classical_prior`
feature plus, depending on `prior_mode`, a logit-space correction), so
signature-generation policy is part of the model's effective input
distribution -- exactly as load-bearing as normalization
(`hydroswarm.runtime.v4_normalization`).

`hydroswarm.training.scout_labels.build_signature_artifact_for_network`'s
own defaults are the REAL policy `data/learning-v2/cycle-b2/signatures/`
(and therefore Stage F's training corpus) were actually fit with:
`start_time_bins=(0, 60, 120, 240)`, `duration_bins=(30, 60, 120)`,
`strength_bins=(0.5, 1.0, 2.0)`. Before this fix,
`hydroswarm.runtime.v4_defaults.V4PipelineFactory.__call__` independently
invented a SMALLER runtime hypothesis grid
(`start_time_bins=(0, 60)`, `duration_bins=(30, 60)`,
`strength_bins=(0.5, 1.0)`) -- a real, provable train/serve policy
mismatch, not a documented design choice. Compounding it,
`SignatureCacheKey.configuration_hash` was a bare version-label string
(`"runtime-signatures-v4"`) rather than an actual hash of the policy that
produced it, so two runs with the same label but a DIFFERENT policy could
collide in `SignatureCache` and silently serve a stale, mismatched
artifact.

`GOVERNED_TRAINING_SIGNATURE_POLICY` below is the real training policy,
`SignaturePolicy.policy_hash` is a real fingerprint over every field that
determines the resulting hypothesis-space signature artifact (fed into
`SignatureCacheKey.configuration_hash`, so a policy change is now
provably a different cache key), and `resolve_signature_mode` implements
the two required explicit modes:

- `GOVERNED_KNOWN_NETWORK`: the served network's content identity
  (`hydroswarm.data.scenarios.network_sha256`, which hashes node/link
  connectivity AND per-link length/diameter/roughness) matches one of the
  exact, PRISTINE (pre-scenario-randomization) base networks this
  policy's training corpus was built from (`KNOWN_TRAINING_TOPOLOGY_HASHES`
  -- `hydroswarm.simulation.build_wntr_network()`,
  `data/topology-transfer/branched-loop.inp`,
  `data/topologies/loop-grid.inp`, matching
  `scripts/generate_cycle_b_corpus.py`'s own `TOPOLOGY_BUILDERS` before any
  per-scenario roughness/demand randomization is applied). A
  runtime-generated artifact for this exact network is directly
  policy-comparable to the real training-owned one (though, per this
  module's own docstring above, still not guaranteed byte-identical unless
  loaded from the literal committed training cache; see
  `hydroswarm.classical.signature_registry` for that stricter guarantee
  where it has been wired).

  IMPORTANT: `network_sha256` includes per-link roughness/length/diameter,
  which `hydroswarm.training.corpus` deliberately RANDOMIZES per individual
  scenario (`GeneratedScenario.manifest.network_sha256` is scenario-specific,
  not family-level -- see `hydroswarm.training.corpus.scenario_to_example`'s
  own `topology_hash=scenario.manifest.network_sha256`). This means any
  individual randomized training scenario's network -- or a frozen
  regression fixture like `data/frozen/golden_network.inp`, itself a single
  randomized snapshot per `data/frozen/manifest.json`'s
  `hydroswarm.evaluation.golden.freeze_golden_inputs` generator -- will
  correctly resolve to `RUNTIME_GENERATED_IMPORTED_NETWORK` here, NOT
  `GOVERNED_KNOWN_NETWORK`, despite belonging to the same topology family.
  This is a deliberate, conservative choice: claiming "governed" provenance
  for a network this exact-hash check cannot actually verify would be the
  false-equivalence problem this section exists to prevent. Recognizing an
  entire randomized family (not just its one pristine base network) is
  real future work, tracked here rather than silently assumed solved.
- `RUNTIME_GENERATED_IMPORTED_NETWORK`: an operator-supplied network this
  policy's training corpus never saw. The resulting artifact is
  deterministically reproducible under this same governed policy, but
  MUST NOT be presented as equivalent to a training-owned artifact --
  callers must record and surface the mode, never silently conflate the
  two.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "GOVERNED_TRAINING_SIGNATURE_POLICY",
    "KNOWN_TRAINING_TOPOLOGY_HASHES",
    "SignatureMode",
    "SignaturePolicy",
    "resolve_signature_mode",
]

SignatureMode = Literal["GOVERNED_KNOWN_NETWORK", "RUNTIME_GENERATED_IMPORTED_NETWORK"]


@dataclass(frozen=True, slots=True)
class SignaturePolicy:
    """A complete, versioned description of how a signature hypothesis
    space is generated -- every field that changes the resulting
    `SignatureArtifact` (and therefore the `classical_prior` feature
    HydroCore consumes) must live here, not as an ad-hoc argument at a
    call site."""

    policy_version: str
    #: Deterministic rule for which nodes are source-hypothesis candidates
    #: (matches hydroswarm.training.corpus.scenario_to_example's own
    #: "every junction is a source candidate" convention).
    source_candidate_definition: str
    start_time_bins: tuple[int, ...]
    duration_bins: tuple[int, ...]
    strength_bins: tuple[float, ...]
    demand_regimes: tuple[str, ...]
    sample_times_seconds: tuple[int, ...]
    #: Deterministic rule for which nodes are modeled as possible sensor
    #: locations (NOT which sensors are actually deployed for a given
    #: incident -- that per-incident subset is a runtime masking concern,
    #: handled by HybridInferencePipeline._signature_observations'
    #: observation_mask, not by this policy). "all_junctions_as_sensor_
    #: candidates" documents the deliberate choice explicitly rather than
    #: leaving "all junctions" and "all deployed sensors" silently
    #: conflated with no named policy at all.
    sensor_layout_policy: str

    @property
    def policy_hash(self) -> str:
        payload = {
            "policy_version": self.policy_version,
            "source_candidate_definition": self.source_candidate_definition,
            "start_time_bins": list(self.start_time_bins),
            "duration_bins": list(self.duration_bins),
            "strength_bins": list(self.strength_bins),
            "demand_regimes": list(self.demand_regimes),
            "sample_times_seconds": list(self.sample_times_seconds),
            "sensor_layout_policy": self.sensor_layout_policy,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


#: The real policy data/learning-v2/cycle-b2/signatures/ (and therefore
#: Stage F's joint-v4 training corpus's classical_prior feature) was
#: actually fit with -- see hydroswarm.training.scout_labels.
#: build_signature_artifact_for_network's own default arguments, which
#: this constant mirrors exactly so a future drift between the two is a
#: single-source-of-truth diff, not two independently-maintained literals.
GOVERNED_TRAINING_SIGNATURE_POLICY = SignaturePolicy(
    policy_version="hydroswarm-signature-policy-v1",
    source_candidate_definition="all_junctions",
    start_time_bins=(0, 60, 120, 240),
    duration_bins=(30, 60, 120),
    strength_bins=(0.5, 1.0, 2.0),
    demand_regimes=("nominal",),
    sample_times_seconds=(0, 3600, 7200, 10800, 14400, 18000, 21600),
    sensor_layout_policy="all_junctions_as_sensor_candidates",
)

#: hydroswarm.data.scenarios.network_sha256(...) of the exact three
#: topologies scripts/generate_cycle_b_corpus.py builds cycle-b2's train
#: split from: golden-reference (hydroswarm.simulation.build_wntr_network()),
#: branched-loop (data/topology-transfer/branched-loop.inp), loop-grid
#: (data/topologies/loop-grid.inp). Computed directly from those three
#: network sources (see this module's own test suite for the
#: reproduction), not hand-copied from a report.
KNOWN_TRAINING_TOPOLOGY_HASHES: frozenset[str] = frozenset({
    "628a6dccfeff1af5a81a41d7374f8408085611ddf5ac925ff01e7b809c89464e",  # golden-reference
    "0b1817cd6c28d42f98b1a1a74cb0234d619ee2985b1c7cf70cba4f274094b056",  # branched-loop
    "0e9cfc042e0876f34a8ecbf9435bcbee3c2d840462a274e5ca831c3b40e4fe88",  # loop-grid
})


def resolve_signature_mode(
    topology_hash: str, known_training_topology_hashes: frozenset[str] = KNOWN_TRAINING_TOPOLOGY_HASHES
) -> SignatureMode:
    """Which of the two required explicit modes a served network falls
    under -- pure lookup, never triggers a simulation or claims byte
    equivalence with a training-owned artifact on its own (see this
    module's docstring)."""

    return (
        "GOVERNED_KNOWN_NETWORK"
        if topology_hash in known_training_topology_hashes
        else "RUNTIME_GENERATED_IMPORTED_NETWORK"
    )
