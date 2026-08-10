# Evaluation Protocol

> **HydroCore-v4's own measured evaluation** is in
> [Final system](FINAL_SYSTEM.md) and
> [reports/results/v4/phase13-metrics-and-baselines.md](../reports/results/v4/phase13-metrics-and-baselines.md).
> HydroCore-v4's own locked final evaluation has **not** been opened
> (`locked_test_opened: false` in
> [architecture-freeze.json](../reports/results/v4/architecture-freeze.json)). The
> "Executed learned matrix" section below documents the prior (S/M/L generation)
> architecture's own, already-completed locked test -- a real, historical result for that
> superseded generation, not a description of v4's (still-unopened) locked evaluation.

Report distributions and failure cases, not only averages.

| Capability | Required measurements |
|---|---|
| Localization | top-1, top-3, MRR, conformal coverage, set size, calibration error |
| Sampling | information gain/sample, candidate reduction, samples to resolution, delay/cost |
| Plans | validity/rejection rate, exposure, service, min pressure, violation minutes, regret |
| Reliability | sensor-fault quality, OOD/abstention quality, unsafe non-abstention, recovery |
| Scale | latency, peak RAM, cache hit rate, disk and runtime by network size |

Baselines are classical-only, neural-only, shared HydroCore, specialist adapters, and hybrid
fusion. Required ablations include static versus dynamic graphs, fixed versus active
sampling, verifier removed versus enabled, uncertainty control removed versus enabled, and
seen versus completely held-out networks. Removing the verifier is evaluation-only; it
must never be a product runtime option.

## Executed learned matrix

`reports/results/medium-evaluation-final.json` performs the locked 200-incident,
2,000-bootstrap comparison across classical, HydroCore-M neural, HydroCore-M hybrid,
HydroCore-S hybrid, and HydroMono-S. The original test tensor hash is checked before use.
No test result enters training or checkpoint selection. M calibration is refit only on the
160-scenario calibration split. M neural reaches 94.5%, M hybrid 94.0%, S hybrid 96.0%,
HydroMono-S 94.5%, and classical 91.5%. M therefore remains unpromoted.

`reports/results/topology-transfer-m.json` evaluates 70 new incidents on a genuinely
different seven-junction branched-loop EPANET graph without neural fine-tuning. Classical,
M neural, and M hybrid reach 35.7%, 47.1%, and 44.3%. Coverage is 27.1% with mean set size
0.41. The unseen topology hash forces `CAUTION` and planning suppression; the experiment
supports the safety boundary rather than a transfer-performance claim.

The golden end-to-end scenario should begin with a broad candidate set, select an
informative sample, show contraction, reject an unsafe plan due to a simulated constraint,
verify an alternative, compare both to no response, explain evidence changes, require
operator approval, and replay the immutable event sequence with Wi-Fi disabled.
# Runtime profiling

`python scripts/benchmark_performance.py` records eager FP32 CPU latency, process RSS,
safetensors disk size, deterministic output hashes/equivalence, and a 1,000-node canonical
model stress case. Its seeded random weights make it a runtime baseline only; accuracy is
measured only with governed checkpoints. ONNX/OpenVINO/INT8 status is explicit in the JSON
and never inferred from eager PyTorch performance.
