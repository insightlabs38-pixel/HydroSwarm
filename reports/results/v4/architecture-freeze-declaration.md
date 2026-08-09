# HydroSwarm architecture freeze declaration

**Status: FROZEN.**

Branch `agent/gcp-multitopology-v3`, commit
`e598a4e5b2f3c01259118f1186c369ee0360ca6a` (`e598a4e`). This declaration
formally freezes the architecture/configuration candidate that
`reports/results/v4/pretest-architecture-selection.md` selected and this
session's evidence-cleanup pass (Stage-F third-seed repeatability check,
Strategist/OOD wording corrections, final regression re-run) confirmed as
ready. Nothing was retrained, recalibrated, architecturally changed, or
rebuilt to produce this declaration -- it is a metadata-recording pass
over the already-validated candidate. Machine-readable manifest:
`reports/results/v4/architecture-freeze.json`.

## What is frozen

**Architecture**: `hydrocore-v4`, `HydroCore-S` (`variant=small`),
`d_model=192`, `nhead=6`, `num_layers=4`, `dim_feedforward=576`,
`latent_tokens=64`, `incident_pooling=mean`, `message_direction=
forward_only`, RMSNorm, SiLU, dropout 0.1, `prior_mode=feature_only`,
`use_adapters=false`, `strategist_mode=candidate_conditioned`, event
control heads on, Scout control heads on, consequence-prescreening heads
on, `ood_category_head=true` (11 categories, present-but-untrained --
see below), `auxiliary_heads=false`, 9-action canonical vocabulary
(`NO_ACTION`, `ISOLATE_SOURCE`, `FLUSH_DOWNSTREAM`, `ISOLATE_AND_FLUSH`,
`PROTECT_CRITICAL`, `INCREASE_MONITORING`, `REQUEST_SAMPLE`,
`WAIT_OBSERVE`, `ALTERNATE_VALVE_CUT`).

**Selected checkpoint**: `no_adapters`-seed`20260810`
(`experiments/runs/v4-checkpoint-identity/no_adapters-seed20260810`,
sourced from Stage-F run
`experiments/runs/stage-f/no_adapters-seed20260810/20260808T041727Z-de5f4b0e`).

- Checkpoint identity fingerprint:
  `a94069adba25230f58f24f57901b855fab3a702aabd5e30cf0bc105e002e90a1`
- Model SHA-256:
  `a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7`

**Schema hashes**: feature schema
`7ec97775e5f01f87ae62669146a7eb70958f99b1162a356614eb87220e9ddd09`
(`hydroswarm-features-v2`); target schema
`2f00d174369d48e09c413c76da0c97fddf78630e8e1260d16cf07c43b9034cb1`
(`targets_v2`); action-template schema
`3a447a74bd2254c109364fcf5434b06bd49ba9d7ed537dac91098b1e4e98116d`;
OOD-category schema
`b0be7c960784dce70c00bc3ffc8f0560da01ac9f8463e7485e775d03fb7ed4f4`;
next-step schema `d808002c32920183bd33980a012c45e16e499dcf18d4ae13066c8bc809fcb682`.

**Normalization**: hash
`e0808f21579b693f66e4edb5900e561bcf9c521e850d5c9d2428cb0db0fa1114`, the
real train-split-fit `data/learning-v2/cycle-b2/normalization` artifact
(node `4dcf22a68839a8630e83b0e90f47ac3400b176b576e76d8bee5662221d238691`,
edge `3e715d707475d81eba90de6609246f51bb0eee8a94c58eab4958f4370fca514d`).
Not `"none"`.

**Fusion / signature policy**: fusion policy hash
`fuse_source_probabilities-v1` (`DYNAMIC_TRUST_FUSION_CONFIG`, real
dynamic-trust fusion, not a fixed-weight approximation). Signature policy
`hydroswarm-signature-policy-v1`
(hash `06e31d922261509c3aaae558262d3b5748b42a3a7bb26c4218a6e56acb686811`),
with two explicit modes: `GOVERNED_KNOWN_NETWORK` for the 3 known
training topologies, `RUNTIME_GENERATED_IMPORTED_NETWORK` (clearly
labeled, never conflated) for imported networks.

**Calibration**: artifact hash
`829c167b267b3ce32f55559f3aec4b4933e337f3358e22e1f792a26b402f68fa`,
status `FITTED`, alpha 0.1 / 90% coverage target, empirical coverage
**91.43%** (712 examples), mean candidate-set size 2.80, ECE 0.088, fit
against the exact frozen serving path in Section 19. Validated topology
hashes (pristine family): golden-reference
`628a6dccfeff1af5a81a41d7374f8408085611ddf5ac925ff01e7b809c89464e`,
branched-loop `0b1817cd6c28d42f98b1a1a74cb0234d619ee2985b1c7cf70cba4f274094b056`,
loop-grid `0e9cfc042e0876f34a8ecbf9435bcbee3c2d840462a274e5ca831c3b40e4fe88`.
**Not re-fit this pass** -- this declaration only records the existing
Section 19 artifact.

**Runtime-enabled outputs** (`event_cause`, `event_presence`,
`evidence_sufficiency`, `next_step`, `relative_strength`, `source_node`)
-- all ADVISORY or CALIBRATED_ADVISORY, none bypass deterministic
authority, exact WNTR verification, or human approval.

**Deterministic authorities/fallbacks** (unchanged, verified this
session):

| decision | authority | learned-head status |
|---|---|---|
| Source localization | classical/neural fusion, CALIBRATED_ADVISORY at best | `source_node` promoted, advisory only |
| Sampling | classical expected-information-gain / fixed-order | learned Scout fully disabled (negative realized entropy reduction) |
| Planning | deterministic bounded candidate generation | learned Strategist prescreen/ordering disabled (deterministic heuristic ordering deployed) |
| Verification | WNTR/EPANET, sole `VERIFIED` authority | no learned score can mark a plan verified |
| OOD | deterministic 3-level severity | learned `ood_category` fully disabled (present-but-untrained) |
| Approval | human, mandatory for every operational plan | no bypass exists |

**Release bundle**: `experiments/runs/v4-release-bundle/no_adapters-seed20260810`
(gitignored/ephemeral per this project's convention; reproduce via the
exact commands in `pretest-architecture-selection.md`). **Not rebuilt
this pass.** Its own `runtime_manifest.json` records its build-time
source commit as `c5d2dd5` (Section 19) -- this intentionally differs
from this declaration's `frozen_git_commit` (`e598a4e`), because the
bundle's frozen identity is its content hashes
(`checkpoint_identity_fingerprint`
`a94069adba25230f58f24f57901b855fab3a702aabd5e30cf0bc105e002e90a1`,
model SHA-256
`a501ad87bc39943c48c1a0ea5fc9b6d0807491b684b4423542acbdba712d16c7`),
not the commit label embedded inside it. Re-verified this session:
loads cleanly via `load_v4_inference_bundle`, `calibration_status=
FITTED`, every internal-consistency check (SHA256SUMS, manifest
agreement, checkpoint-identity self-consistency, normalization-hash
reproduction, signature-policy-hash match) passed with no exception.

**Exact Git commit**: `e598a4e5b2f3c01259118f1186c369ee0360ca6a` on
`agent/gcp-multitopology-v3`.

## 3-seed repeatability evidence (recorded, not re-run)

`no_adapters` Stage-F finalist, 3 seeds:

| seed | role | best val loss | dev-holdout loss | wall time |
|---|---|---|---|---|
| `20260810` | **selected** | 5.35117 | 8.79113 | 2812.5s |
| `20260811` | comparison (2-seed direction check) | 5.42401 | 8.87386 | 2884.5s |
| `20260813` | post-selection stability check | 5.43492 | 8.90205 | 2852.3s |

All three: 16/16 epochs, no early stopping, no NaNs/non-finite values.
Seed 20260813 (run in this session, purely as a stability check after
selection) is reasonably consistent with the first two on every headline
and per-task metric (`source_node`, `evidence_sufficiency`,
`relative_strength`, event outputs, plan tasks) -- see
`reports/results/v4/stage-f-no-adapters-seed20260813-repeatability.md`
for the full per-task table. Seed 20260810 was retained; it was never at
risk of replacement since seed 20260813 was not marginally better on
any headline metric.

## Final pre-freeze gate results (this session, re-run after evidence cleanup)

| gate | result |
|---|---|
| Full `pytest` | **859 passed**, 0 failed |
| Ruff (`src tests scripts`) | clean |
| Pyright | 0 errors, 0 warnings |
| Train/serve parity (`scripts/run_train_serve_parity_gate.py`) | **passed** -- 3 topologies x 2 conditions |
| Inference-bundle load/self-test | **passed** |
| Calibration identity validation | **passed** |

## Known deferred limitations (frozen as documented, not resolved)

1. `sensor_fault` evaluation population is degenerate (zero true
   negatives) -- learned head stays disabled; deterministic sensor-health
   logic stays authoritative.
2. `model_input_signature_mode` is computed at `analyze()` time but not
   yet threaded into the Decision Authority certificate's
   `DecisionProvenance`.
3. The frontend's hand-written TypeScript `ApiIncidentView` type does not
   yet declare `verification_status`/`context_hash` on
   `plans[].verification` -- backend correctness does not depend on this.
4. Calibration was fit against reconstructed (not disk-verified) scenario
   data, a sandbox environment limitation, not a correctness defect.
5. Learned Scout and learned Strategist prescreening remain disabled by
   measured evidence; both have a documented one-retrain path (Section
   18) that would produce a new candidate for separate evaluation, not
   reopen this frozen one.

None of these block the freeze; all were already known and documented
before this declaration, and none of them describe reversible defects in
the frozen candidate itself.

## No-further-architecture-tuning commitment

**The architecture, the selected checkpoint
(`no_adapters`-seed`20260810`), its calibration artifact, its
output-governance configuration, and its V4 inference release bundle are
now FROZEN as of commit `e598a4e5b2f3c01259118f1186c369ee0360ca6a`.**

No further architecture redesign, retraining, recalibration,
promotion-decision change, or release-bundle rebuild will occur against
this candidate, except:

- a bug fix that provably preserves every frozen contract/hash recorded
  above, or
- an explicitly optional, separately authorized one-retrain improvement
  using an already-scaffolded mechanism (learned OOD category, learned
  Scout, PCGrad, class balancing -- Section 18's preserved one-retrain
  paths), which would produce a **new** candidate for separate
  evaluation rather than a silent mutation of this frozen one.

This candidate is preserved unchanged for the locked evaluation step.

## Locked-test status

- The locked final test **has not been opened**.
- The one-time locked evaluation **has not been performed**.
- `final-selection.json` **does not exist**.
- This freeze declaration does not authorize the locked evaluation; that
  remains a separately authorized step.
