# Offline Study 1 measurement correction

The original Study 1 artifact was committed at `d167ce6` before this
correction. It remains preserved in Git as blobs:

| Original artifact | Git blob |
|---|---|
| `results.json` | `00742c893ca2039dd3317fd368b462729fe9eb8f` |
| `results.csv` | `5028fc55df5b7ab159ec6eb37e8a56177f8bf387` |
| `summary.json` | `9f71c88e3f7baa93264826a9cf3f0208717a7654` |

The original model-localization values are not reinterpreted. The following
harness-label defects are corrected by a complete deterministic rerun:

- `first_recommended_node` was a highest-posterior source proxy. No actual
  active-sampling engine ran, so it is now null.
- `analysis_ms` and `total_workflow_ms` were populated from model-forward
  timing. They are now null; only `inference_ms` is measured.
- Point-in-time `process.memory_info().rss` was labelled `peak_rss_mb`. It is
  now `process_rss_mb`. Study 2 records peak/high-water memory separately
  when it actually measures a LIVE workflow.
- The prior OOD conclusion is a governed-policy replay. It is not evidence
  that the live runtime independently detected those states.

This correction changes evaluation harness output only. No production source,
weights, calibration, thresholds, corpus, or locked split was modified.
