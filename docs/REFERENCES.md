# References

## Prior art and differentiation

HydroSwarm does not claim that contamination localization, hydraulic simulation, graph learning, conformal prediction, or active sampling are individually novel. Its implemented contribution is the integration and governance of these pieces into an auditable decision-support authority chain: hybrid source evidence, explicit calibration applicability, deterministic OOD/sampling/planning controls, exact response verification, human approval, and local reproducibility.

| Prior system / approach | Relevant idea | HydroSwarm relationship |
|---|---|---|
| EPANET / [WNTR](https://usepa.github.io/WNTR/) | hydraulic and water-quality simulation | used for governed synthetic evidence and exact modeled response verification; HydroSwarm does not claim to replace the simulator |
| [TEVA-SPOT](https://www.epa.gov/water-research/teva-spot-toolkit) | consequence/sensor-placement analysis | informs the importance of modeled consequence and evidence placement; HydroSwarm focuses on an integrated local incident decision workflow |
| [CANARY](https://www.epa.gov/water-research/canary-event-detection-software) | water-quality event detection | relevant monitoring context; HydroSwarm combines a separate source-localization advisory with governed response verification |
| Bayesian source localization ([Dawsey et al.](https://doi.org/10.1061/(ASCE)0733-9496(2006)132:4(234)); [Jerez et al.](https://doi.org/10.1016/j.ymssp.2021.107834)) | uncertain source ranking | motivates probabilistic/classical evidence; HydroSwarm adds a learned residual, applicability/fail-closed controls, and response authority boundaries |
| Grab-sampling localization ([Ji et al.](https://doi.org/10.1029/2022WR032784)) | evidence acquisition for source identification | motivates additional sampling; HydroSwarm's final sampling authority remains deterministic and advisory to a human workflow |
| Graph neural methods for water networks ([Spatial GCNs](https://arxiv.org/abs/2211.09587)) | graph-structured learning | relevant to HydroCore-v5's learned Sentinel; physical verification and operational authority remain outside the network |

The table intentionally makes narrow claims about prior systems rather than asserting that a cited system lacks every HydroSwarm feature.

## Water networks and response

- US EPA, [Water Network Tool for Resilience (WNTR)](https://usepa.github.io/WNTR/).
- US EPA, [TEVA-SPOT Toolkit](https://www.epa.gov/water-research/teva-spot-toolkit).
- US EPA, [CANARY Event Detection Software](https://www.epa.gov/water-research/canary-event-detection-software).
- US EPA, [Information about Public Water Systems](https://www.epa.gov/dwreginfo/information-about-public-water-systems) — public water system count and population served.
- US EPA, [Fact Sheet about Water Quality Surveillance and Response System](https://www.epa.gov/waterresilience/fact-sheet-about-water-quality-surveillance-and-response-system) — monitoring/sampling/response framework.
- University of Kentucky, [Water Distribution System Research Database](https://uknowledge.uky.edu/wdsrd/).

## Models, uncertainty, and training

- Rampášek et al., [Recipe for a General, Powerful, Scalable Graph Transformer](https://arxiv.org/abs/2205.12454), 2022.
- Jaegle et al., [Perceiver IO](https://arxiv.org/abs/2107.14795), 2021.
- Houlsby et al., [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html), 2019.
- Yu et al., [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), 2020.
- Chen et al., [GradNorm](https://proceedings.mlr.press/v80/chen18a.html), 2018.
- Ross et al., [DAgger](https://proceedings.mlr.press/v15/ross11a.html), 2011.
- Angelopoulos and Bates, [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511), 2021.
- Liu et al., [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759), 2020.

## Application stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Vite](https://vite.dev/)
- [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/)

## Project-specific evidence

External literature supplies context, not the final numerical claims. Final HydroCore-v5 identity and measured results are backed by repository-generated artifacts:

- [Finalist freeze](../reports/evaluation/hydrocore-v5/m11/m11-2/finalist-freeze.json)
- [Runtime manifest](../models/hydrocore-v5-release/runtime_manifest.json)
- [M11.6 metrics](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-metrics.json)
- [M11.6 gate](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-gate.json)
- [M11.6 safety counters](../reports/evaluation/hydrocore-v5/m11/m11-6-final/m11-6-safety-counters.json)

External links and licenses should be rechecked for future releases; no unverified external benchmark is used to characterize the final locked V5 result.
