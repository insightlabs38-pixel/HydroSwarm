# Milestone 10.0 summary: final predictor / system preflight

Branch: exp/hydrocore-v5-causal. Commit: 8284e48095acd14270be962250d1aebe9ce5cbcf. locked_test_opened before/after: False/False.

## Predictor identity
- variant: small, 4182612 parameters, checkpoint policy FINAL_STEP_1350
- all 3 seed checkpoints SHA-256-verified against M9.6 canonical record

## Interfaces
- forward-pass output keys: ['candidate_reduction_prediction', 'duration_logits', 'event_cause_logits', 'event_presence_logits', 'evidence_sufficiency', 'expected_information_gain', 'hidden_state', 'latent_state', 'next_step_logits', 'node_mask', 'ood_category_logits', 'ood_logits', 'relative_strength_logits', 'sample_node_logits', 'scout', 'sensor_fault_logits', 'sentinel', 'should_continue_sampling_logits', 'source_node_logits', 'source_region_logits', 'start_time_logits', 'strategist', 'uncertainty']
- source posterior present: True
- evidence-sufficiency present: True
- ood_category head present (trained, jointly with backbone): True
- scout control heads present (raw): True
- strategist role-hidden group present (raw, generic): True
- strategist NAMED candidate-conditioned proxy heads present: False (expected False -- see schema status below)
- conformal candidate set callable: True
- fusion callable, finite: True

## Authority / fallback boundaries
- Scout/Strategist deterministic fallback reachable: True / True
- deterministic OODDetector callable: True
- high-OOD forces ABSTAIN (deterministic control policy): True
- learned ood_category is advisory-only, excluded from runtime_enabled_outputs in every real checkpoint identity built so far: True
- no learned component autonomously actuates: True
- WNTR/EPANET remains downstream authority: True

## Scout / Strategist schema status
`SCOUT_STATE_SCHEMA_VERSION = "scout-state-v1-unbuilt"` and `STRATEGIST_CANDIDATE_SCHEMA_VERSION = "strategist-candidate-v1-unbuilt"` (`src/hydroswarm/training/checkpoint_identity.py`): raw model heads exist and are present in every forward pass, but the full candidate-conditioned contract/schema layer connecting them to `HydroScout`/`HydroStrategist` is not yet built. M10.2/M10.3 (not run by this task) will need a preflight-correction pass before a scientific comparison is executable there.

## Result
**M10_0_RESULT = SYSTEM_PREFLIGHT_PASS.** M10.1 (OOD/fusion) may proceed: both the deterministic OODDetector/fusion path and the neural ood_category head/fusion-feature path are fully implemented and callable today.
