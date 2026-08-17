# M9.1 provenance refresh (2026-08-17)

Amends `docs/evaluation/HYDROCORE_V5_M9_1_PROTOCOL.md` Section 21 (dated 2026-08-17 entry). This is a narrow, audited provenance/preflight refresh, **not** a reopening of M9.1's scientific result: no `development_holdout`, calibration, `locked_final_test`, or `locked_topology_test` data was loaded or inspected at any point, and no M9.1 metric/guardrail/decision/artifact is altered.

## Trigger

After M9.8 closed, the full repository suite reported 1 failure:

```
tests/scientific/test_m9_1_runner.py::test_assert_code_under_test_commit_passes_at_current_head
```

M9.8's own closure commit (`e33d71804c517d712724488ce9f005093cd06cb5`) already disclosed this: M9.1's `assert_code_under_test_commit` correctly and intentionally trips on **any** change under `src/hydroswarm/model/core.py`, and M9.7 (commit `475874d`) made such a change -- registering a new `MODEL_VARIANTS["small_v5_capacity_m"]` entry for the M9.7/M9.8 capacity comparison. The failure is a correct tripwire firing, not evidence that M9.1's ODE/CDE/SDE result became invalid.

## Audit (mechanical, no predictive data)

| Check | Result |
|---|---|
| Commits touching `continuous_time.py`/`core.py`/`training-v5-causal.yaml` between the prior floor (`154605180f2a950d86452cfc8ec7202990aba8cf`) and M9.7's commit (`475874d8977d0952e8fc3626eb2bd6580cc3c2f7`) | exactly one: `475874d` itself |
| `continuous_time.py` diff | empty |
| `configs/training-v5-causal.yaml` diff | empty |
| `core.py` diff | pure addition, 17 lines added / 0 removed |
| `core.py` diff content | one new `MODEL_VARIANTS` entry, `"small_v5_capacity_m"`; `"small"`/`"medium"`/`"large"` untouched |
| `MODEL_VARIANTS` lookup mechanism | strict key lookup (`MODEL_VARIANTS[variant.lower()]`), never iterated -- an added key cannot affect `"small"`'s resolved config |
| `"small"` variant parameter count under M9.1's frozen `SHARED_MODEL_CONFIG`/`CURRENT_MODEL_KWARGS` | 4,182,612 (unchanged, matches `CURRENT_BASELINE_TOTAL_PARAMS`) |
| GRAPH_ODE/CDE/SDE construction | unaffected (lives in `continuous_time.py`, byte-unchanged) |
| Temporal dynamics / training config | unaffected |

Full detail: `m9-1-provenance-audit.json`.

## Fix

`scripts/hydrocore_v5/m9_1_common.py`'s `CODE_UNDER_TEST_COMMIT_FLOOR` re-superseded from `154605180f2a950d86452cfc8ec7202990aba8cf` (preserved as `CODE_UNDER_TEST_COMMIT_FLOOR_V1`) to `475874d8977d0952e8fc3626eb2bd6580cc3c2f7`, via the SAME "later commit that changes nothing under the three frozen paths" mechanism the guard already implements -- not a new mechanism, not a weakened one. Any commit after `475874d` touching any of the three `FROZEN_UNCHANGED_PATHS` still trips the guard exactly as before.

Two new regression tests added to `tests/scientific/test_m9_1_runner.py`:

- `test_code_under_test_commit_floor_v2_audit_is_additive_only` -- re-runs the git-log/git-diff audit above on every invocation, so a future history rewrite that changed the commit range's actual content would fail the test rather than silently pass.
- `test_frozen_small_variant_param_count_unchanged_at_current_head` -- re-confirms the 4,182,612 parameter count at whatever commit the suite runs at.

## Test results

- `tests/scientific/test_m9_1_runner.py`: 82/82 passed (80 pre-existing + 2 new)
- `tests/scientific/test_m9_1_preflight.py`: 47/47 passed (unchanged)
- `tests/scientific/test_m9_7_capacity_preflight.py` + `test_m9_7a_checkpoint_policy.py` + `test_m9_8_capacity_comparison.py`: 84/84 passed (unaffected)
- `pyright`: 0 errors on touched files
- Full repository suite: **1539 passed, 1 skipped, 0 failed** (previously 1536 passed / 1 skipped / 1 failed -- the +2 delta is this entry's own 2 new tests, and the prior failure is now a pass, for zero net regressions)

## Locked-test state

`locked_final_test` / `locked_topology_test`: unopened before and after this entire audit.

## Conclusion

- No M9.1 scientific rerun performed.
- No M9.1 scientific conclusion changed.
- Provenance guard restored and left fully intact for future changes (not weakened to "any future core.py change is fine").
- `test_assert_code_under_test_commit_passes_at_current_head` now passes without any xfail/skip.
