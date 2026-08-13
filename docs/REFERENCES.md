# References

## Prior Art & Differentiation

HydroSwarm does not claim that contamination localization is novel. The
distinction below is the implemented, integrated authority path: calibrated
localization + classical/neural disagreement + active evidence acquisition +
deterministic safety gates + exact WNTR verification + explicit human approval
+ local/offline reproducibility and audit.

| Prior system / approach | What it does | What HydroSwarm uses or learns from | Where HydroSwarm differs |
|---|---|---|---|
| EPANET / [WNTR](https://usepa.github.io/WNTR/) | Hydraulic and water-quality simulation, including resilience analysis tooling | Governed WNTR/EPANET scenario generation and exact response verification | Uses solver output as a required verification boundary inside an authority workflow; it does not claim to replace these simulators |
| [TEVA-SPOT](https://www.epa.gov/water-research/teva-spot-toolkit) | EPA toolkit for consequence assessment and sensor-placement analysis | The importance of simulated consequence/sensor evidence | Different scope: local incident workflow with current calibrated localization, explicit gates, and human approval |
| [CANARY](https://www.epa.gov/water-research/canary-event-detection-software) | Water-quality event detection software | Event-detection and water-quality-monitoring context | HydroSwarm does not represent CANARY as a source-localization or response-approval equivalent; it combines a separate advisory localization path with verification gates |
| Bayesian source localization ([Dawsey et al.](https://doi.org/10.1061/(ASCE)0733-9496(2006)132:4(234)); [Jerez et al.](https://doi.org/10.1016/j.ymssp.2021.107834)) | Combines uncertain monitoring/simulation evidence to rank plausible contamination sources | Classical posterior reasoning and explicit uncertainty | Adds a governed neural residual, calibration applicability checks, disagreement handling, and a response authority boundary; it does not claim Bayesian localization itself is novel |
| Grab-sampling localization ([Ji et al.](https://doi.org/10.1029/2022WR032784)) | Uses manual samples to improve contamination-source identification | The operational value of additional evidence | Uses deterministic expected-information-gain ranking only as an advisory sampling recommendation, with no autonomous field action |
| Graph neural methods for water networks ([Spatial GCNs](https://arxiv.org/abs/2211.09587)) | Apply graph learning to water-system tasks | Graph-structured feature processing | HydroCore-v4 is one advisory branch; exact simulation, deterministic gates, and human approval are not delegated to the network |

Claims above describe the cited systems only at the stated level; they do not
assert that a prior system lacks an unverified feature.

## Water networks and response

- US EPA, [Water Network Tool for Resilience (WNTR)](https://usepa.github.io/WNTR/).
- US EPA, [TEVA-SPOT Toolkit](https://www.epa.gov/water-research/teva-spot-toolkit).
- US EPA, [CANARY Event Detection Software](https://www.epa.gov/water-research/canary-event-detection-software).
- US EPA, [Water Quality Surveillance and Response](https://www.epa.gov/waterresilience).
- University of Kentucky,
  [Water Distribution System Research Database](https://uknowledge.uky.edu/wdsrd/).

## Models, uncertainty, and training

- Rampášek et al., [Recipe for a General, Powerful, Scalable Graph Transformer](https://arxiv.org/abs/2205.12454), 2022.
- Jaegle et al., [Perceiver IO](https://arxiv.org/abs/2107.14795), 2021.
- Houlsby et al., [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html), 2019.
- Yu et al., [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html), 2020.
- Chen et al., [GradNorm](https://proceedings.mlr.press/v80/chen18a.html), 2018.
- Ross et al., [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), 2011.
- Angelopoulos and Bates, [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511), 2021.
- Liu et al., [Energy-based Out-of-distribution Detection](https://arxiv.org/abs/2010.03759), 2020.

## Application stack

- [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/),
  [Vite](https://vite.dev/), [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/),
  and [OpenVINO model conversion](https://docs.openvino.ai/nightly/openvino-workflow/model-preparation/convert-model-pytorch.html).

URLs were last reviewed for the project plan on 2026-08-03. Dataset licenses and software
versions must be rechecked when preparing a release.
