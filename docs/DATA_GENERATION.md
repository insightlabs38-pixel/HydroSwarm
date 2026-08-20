# Data generation and governance

HydroSwarm's scientific evidence is generated from governed WNTR/EPANET trajectories and manifests, not ad-hoc feature tables. The final V5 program separates scenario generation, causal-prefix views, training, calibration, development evaluation, and the one-time locked population.

## Current V5 workflow

1. **Define physical scenario/topology identity.** Record network/topology hashes, seeds, simulator inputs, source labels, timing, and corruption/condition configuration.
2. **Assign split role before derived views.** Training/validation/calibration/development/locked roles are fixed before causal prefixes or augmentations are generated.
3. **Generate causal evidence views.** Prefix depths are views over an already-split physical scenario; future observations are excluded.
4. **Train only on authorized training data.** The final S predictor was trained on 600 physical scenarios across three interleaved topology families.
5. **Fit calibration only on calibration-role evidence.** Calibration is separate from weight/checkpoint optimization and is bound to model/feature/fusion identity.
6. **Use development populations for architecture/capability/trajectory work.** M9/M10 comparisons occur before finalist freeze.
7. **Freeze finalist and full pre-lock validation.** M11.2 freezes exact identity; M11.5 must be green.
8. **Freeze and materialize the locked population without evaluation.** The materialization manifest records seed/topology novelty and overlap controls while `locked_test_opened=false`.
9. **Require explicit one-time authorization.** The OPENED record precedes the only M11.6 execution.
10. **Never tune or rerun after lock.** Terminal governance records `locked_rerun=false` and `post_locked_tuning=false`.

## Mechanically verified V5 artifacts

- [M1 causal-prefix population](../reports/evaluation/hydrocore-v5/m1-prefix-dataset.json)
- [M9.6 final training record](../reports/evaluation/hydrocore-v5/m9-6/m9-6-training-runs/ARM_B_M9_6-seed20260814.json)
- [M9.6 evaluation manifest](../reports/evaluation/hydrocore-v5/m9-6/m9-6-manifest.json)
- [M10.4 full-trajectory population](../reports/evaluation/hydrocore-v5/m10/m10-4/m10-4-population-manifest.json)
- [M11.2 finalist freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json)
- [M11.6 materialization manifest](../data/locked/m11-6/m11-6-materialization-manifest.json)
- [M11.6 opened record](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-opened-record.json)

## Final-training facts

The selected `ARM_B_M9_6` run is a deterministic CPU training record with:

- `small` HydroCore-v5, 4,182,612 parameters;
- 600 physical training scenarios;
- 200 each from `golden-reference`, `branched-loop`, `loop-grid`;
- equal family weighting;
- four microbatches per optimizer update;
- exactly 1,350 optimizer steps;
- canonical policy `FINAL_STEP_1350`;
- selected seed `20260814`;
- canonical checkpoint SHA-256 `de2b3f...d7b6d2a5`.

The training config declares more head weights than the corpus validly supervises. The final governance correctly resolves that ambiguity by freezing the trained task family as `sentinel` only.

## Locked generation is intentionally different

The M11.6 locked population is not an ordinary “regenerate and rerun” dataset. Its design was frozen before materialization, its seed namespace was isolated from prior work, four novel topologies were generated/audited, and the resulting population was opened exactly once after authorization.

Do **not** use M11.6 as a routine reproduction command. See [Reproducibility](REPRODUCIBILITY.md).

## Validation expectations

Generated artifacts should carry, where applicable:

- generator/evaluator identity and hashes;
- topology/network hashes;
- seed and seed-domain information;
- split role;
- feature/schema/model/calibration identity;
- population counts and completeness;
- finite-value/schema checks;
- overlap/novelty checks;
- explicit caveats when a comparison cannot be mechanically established.

## External networks

HydroSwarm can import EPANET `.inp` files for local runtime use. External/public network sources have their own licensing and provenance requirements; the repository should not treat availability as permission to redistribute.

## Historical / legacy generation workflow

> **Historical record.** The commands below describe the earlier learning-v1/S-M-L workflow and are retained only for reproducibility of that generation. They are not the final V5 training/evaluation recipe.

```powershell
python scripts/prepare_training_corpus.py --output data/learning-v1 --train-count 800 --validation-count 160 --calibration-count 160 --test-count 200
python scripts/rebuild_canonical_tensors.py
python scripts/train.py --config configs/training_benchmark.yaml --train-manifest data/learning-v1/tensors-canonical-v3/train.jsonl --validation-manifest data/learning-v1/tensors-canonical-v3/validation.jsonl
python scripts/evaluate_learning.py --help
```

For current V5 provenance, prefer the frozen M9/M10/M11 artifacts linked above over reconstructing a command from historical documentation.
