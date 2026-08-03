# Evaluation Protocol

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

`reports/results/learning-evaluation-final.json` evaluates 200 withheld hydraulic-regime
incidents with 2,000-sample bootstrap confidence intervals. Classical-only, HydroCore-S,
HydroMono-S, hybrid fusion, and partial HydroCore-M share the same scenarios and source
budget. Hybrid reaches 96.0% top-1 versus 91.5% classical; HydroCore-S and HydroMono-S tie
at 94.5%. Conformal calibration uses a separate 160-scenario split and reaches 91.0%
held-out coverage at alpha 0.1. The result does not cover independent-topology transfer.

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
