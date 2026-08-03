# HydroSwarm
## Offline Neuro-Hydraulic Incident Intelligence

Technical report - Reverie Hacks 2026 Software Development submission

Version 0.1.0 | August 2026

## Abstract

Drinking-water quality alerts are difficult to interpret because observations are sparse,
hydraulic direction changes over time, sensors fail, and interventions can create pressure
or service failures. HydroSwarm is an offline decision-support system that combines
hydraulic state reconciliation, time-directed graph screening, Bayesian source signatures,
a graph and time-series neural model, calibrated uncertainty, active sampling, and exact
WNTR verification. It produces typed evidence and operational alternatives rather than
autonomous commands. A frozen WNTR regression scenario demonstrates the complete loop:
four initially plausible sources, a measured one-bit sample selection, candidate
contraction, rejection of an unsafe closure, verification of a safer flush alternative,
comparison against no response, a human approval pause, and deterministic replay. Results
are simulation measurements and are not claims of field performance.

## 1. Problem and motivation

A single abnormal observation rarely identifies where an incident began. Water moves
through a network whose effective direction depends on demand, storage, pumps, valves, and
time. Several upstream locations can produce similar observations, while a frozen or
mis-scaled sensor can imitate a network event. Field sampling is slow and costly. Response
actions also couple public-health objectives to hydraulic constraints: isolation or
flushing can reduce modeled exposure yet cause low pressure or unserved demand.

HydroSwarm addresses the interval between detection and an operator decision. It is built
for incident commanders, hydraulic engineers, water-quality staff, and sampling teams. It
does not identify chemistry, certify potability, issue public-health advice, connect to
SCADA, or execute infrastructure changes.

## 2. Related tools and methods

WNTR provides the authoritative EPANET-based hydraulic and quality simulation layer.
NetworkX supports explicit graph algorithms. The learned architecture is informed by
GraphGPS-style local and global processing, Perceiver-style latent attention, and
parameter-efficient adapters. Split conformal prediction supplies empirical candidate-set
coverage, while energy and disagreement signals contribute to out-of-distribution
detection. Expected information gain connects uncertainty to the next field sample.

The central design decision is hybrid rather than purely neural: a source-signature model
remains inspectable and operational when no checkpoint is present, and exact simulation is
the final authority for response-plan status.

## 3. System architecture

Telemetry and samples first enter state reconciliation. A hydraulic run produces the
time-specific directed graph. Classical feasibility and signature likelihood operate in
parallel with HydroCore. Fusion trust changes with sensor health, disagreement,
calibration validity, and OOD evidence. The resulting posterior controls either active
sampling or plan generation. Every plan passes deterministic constraints and exact WNTR
consequence analysis before it can be labeled VERIFIED. A person must then approve it;
HydroSwarm has no action-execution interface.

FastAPI serves typed JSON, WebSocket updates, background-job progress, and the built React
console from one local process. SQLite persists networks, incidents, evidence, posterior
history, jobs, plans, approvals, and a hash-chained event ledger. Runtime modes degrade
from full hybrid through degraded hybrid and classical safe to simulation-only.

## 4. Classical inference

State reconciliation adjusts demand multipliers, tank state, and pump or valve status
within bounded uncertainty and reports residual mismatch. The dynamic graph orients links
using simulated flow at the relevant time; near-zero and reversed flows remain explicit.
Feasibility screening removes sources inconsistent with directed travel time and sample
timing.

The source-signature service enumerates source node, start time, duration, strength, and
demand regime. Exact quality simulations produce sensor-time response tensors. Compressed
NPZ artifacts carry a complete cache key and checksum; mismatch or corruption forces an
exact rebuild. A Gaussian residual likelihood and prior produce hypothesis probabilities,
which are aggregated to nodes and connected candidate regions. Explanations include
residual magnitude and physical exclusion reasons.

## 5. HydroCore architecture

HydroCore receives canonical 19-dimensional node features, 13-dimensional edge features,
temporal sensor histories, observation masks, and sensor-quality features. Local
edge-aware graph processing captures adjacency and direction. A fixed latent bottleneck
provides global attention at bounded cost. RMS normalization, SiLU activations, residual
connections, and dropout support stable training.

Semantic heads predict source logits, incident timing and strength, sensor health,
sampling value, typed plan actions, consequence residuals, and fixed explanation intents.
Small, medium, and large configurations contain approximately 4.0M, 12.4M, and 24.4M
parameters. HydroMono and no-adapter variants support ablation. Missing checkpoints are
reported as not run; the application never substitutes fabricated model results.

## 6. Specialist agents and control

HydroSentinel validates evidence, assesses sensor versus network events, and localizes a
source region. HydroScout ranks accessible, non-duplicate candidates by expected posterior
information gain, likely region reduction, detection probability, delay, cost, and
redundancy. HydroStrategist creates diverse bounded templates including no response,
flushing, isolation, valve and pump adjustments, and public-notification recommendations.
HydroVerifier alone can promote a plan after exact simulation.

An 18-state deterministic controller owns permissions, timeouts, retry budgets, and human
checkpoints. Malformed agent output is rejected. Sample arrival triggers a new posterior
revision rather than mutating history. Idempotency and replay tests cover repeated events
and recovery.

## 7. Data generation and governance

The WNTR scenario generator randomizes source, timing, duration, strength, demand, tank
level, roughness, outages, sensor layout, noise, drift, freezing, communication loss,
jitter, quantization, unit mismatch, and flow reversal. Curriculum stages progress from
clean through operational, degraded, distribution-shift, and adversarial conditions.

Network-disjoint split ownership is assigned before simulation. Manifests record seeds,
network and artifact hashes, simulator and generator versions, physical settings, sensor
layout, corruption settings, replay hash, and provenance. Validators enforce finite values,
aligned time axes, correct missing-value masks, replay consistency, and seed-family
separation. Calibration and final test partitions are never used for model fitting.

## 8. Training

Training uses governed JSONL trajectories and PyTorch. Multitask semantic losses cover
localization, event parameters, sensor health, sampling, actions, consequences, and
explanation intents. AdamW, warmup scheduling, gradient accumulation, clipping, mixed
precision where supported, early stopping, task-weight logging, optional GradNorm, and
optional PCGrad are available. Runs record the resolved configuration, git state, dataset
manifest hash, seeds, environment, progress, failures, and resource guards.

Checkpoints are safetensors with resume state and checksums. Training can resume after a
bounded interruption. A language decoder, when used, is trained against fixed evidence
intents with the operational latent detached so fluent text cannot alter scientific heads.

## 9. Uncertainty, OOD, and abstention

Split and Mondrian conformal artifacts are fit only on held-out calibration examples and
versioned against model, feature schema, and dataset manifest hashes. Reports include
coverage, mean set size, expected calibration error, and condition/network breakdowns.
Invalid artifacts fail closed.

OOD assessment combines feature-range departure, missingness, energy score, classical and
neural disagreement, and physical residuals. Severe OOD or invalid calibration suppresses
confident planning. Dynamic trust favors classical evidence under sensor/model conflict and
neural residual correction only within validated conditions. Abstention is visible to the
operator and changes the controller path.

## 10. Plan verification and consequences

Plans are typed action sequences with targets, values, units, timing, constraints, and
rationale references. Deterministic checks limit action count, targets, ranges, conflicts,
and exact-simulation budget. WNTR evaluates pressure, demand delivery, tank behavior,
quality transport, exposure mass and volume, population and pipe-extent proxies, pump
energy, and service consequences. Timeouts, instability, missing output, or partial runs
are rejections.

Plans are compared with a separately simulated no-response branch. Multi-objective ranking
reports Pareto status and regret instead of hiding tradeoffs in one scalar. VERIFIED means
only that all configured simulation constraints passed; APPROVED is a separate recorded
human decision and never triggers physical execution.

## 11. Evaluation protocol

The evaluation framework measures localization top-k and true-source probability,
candidate contraction, entropy reduction and information gain, unsafe rejection, safe
acceptance, approval pause, deterministic replay, exposure reduction, latency, Python
allocation peak, and cache behavior. Repeated seeded runs produce confidence intervals.
Baselines and ablations include uniform localization, no active sampling, no exact
verifier, and no cache. Neural variants are evaluated only when compatible trained
checkpoints are supplied.

The frozen golden network, scenario, and manifest are checksummed. All displayed outcomes
are regenerated through WNTR. Runtime is excluded from the reproducibility hash, while
scientific inputs and results remain included.

## 12. Measured results

The checked-in regression report passes its promotion gate. A uniform four-node prior
contracts to the true J2 source at approximately 99.4 percent after sampling J2. The sample
has 1.0 bit measured information gain. WNTR rejects a sole-feeder closure for pressure
below the minimum and verifies a J4 flush alternative with full modeled service. Separate
quality runs estimate 2,541,416 mg no-response consumption and 2,526,693 mg for the verified
alternative, a reduction of 14,723 mg. The controller pauses for both sampling and human
approval and completes deterministic hash-chain replay.

These values are a compact regression proof, not a population estimate and not evidence
of real-world accuracy. The authoritative machine-readable results are included alongside
this report.

## 13. Ablations and failure cases

Removing exact verification fails the promotion gate by design: a proposal cannot become
VERIFIED. Disabling the signature cache is actually executed and increases repeated-run
latency. Removing active sampling leaves the broad source set unchanged. The learned-model
ablations remain explicitly not run until governed checkpoints are available.

Important failure cases include non-identifiable sensor layouts, stale hydraulic state,
unknown controls, cache mismatch, unit error, timing jitter, frozen or drifting sensors,
simulation timeout, numerical instability, and distribution shift. Each produces either a
visible warning, broader uncertainty, fallback, abstention, or rejection rather than a
silent success label.

## 14. Security, reliability, and deployment

The native application binds to loopback, performs no runtime internet calls, accepts no
URLs, validates and hashes INP files, limits request and job sizes, and stores data only in
its configured directory. The container publishes loopback only, drops capabilities,
prevents privilege escalation, uses a read-only root filesystem, and persists a dedicated
data volume. Self-test executes real inference, WNTR, SQLite, dependency, resource, port,
and frontend checks.

The React interface is API-first and labels its deterministic fixture fallback. It provides
a tile-free operational map, topology view, hydraulic charts with text equivalents,
timeline replay, evidence-change display, synchronized plan branches, audit and validation
views, keyboard navigation, visible focus, reduced motion, and non-color status cues.

## 15. Limitations

HydroSwarm has not been validated on live utility incidents. Synthetic and reference-network
evaluation cannot establish field safety, chemical identity, toxicity, pathogen viability,
regulatory compliance, or public-health outcomes. WNTR predictions inherit topology,
demand, control, mixing, and sensor-model errors. Population exposure is a proxy unless
governed demographic and consumption data are supplied. The API is not an authenticated
internet-facing multi-tenant service.

No output authorizes flushing, isolation, equipment changes, notification, or any other
field action. Qualified utility and public-health personnel must follow current procedures,
laboratory evidence, regulations, and engineering review.

## 16. Reproducibility

Python dependencies are frozen by uv; the exported runtime set is hash-locked and audited.
The React dependency lock is vulnerability-audited. CI covers Windows and Ubuntu, Python
tests with branch coverage, Ruff, Pyright, self-test, wheel build, frontend lint, type/build,
component accessibility tests, and Playwright. The golden scenario and evaluation scripts
write JSON and Markdown outputs with stable scientific hashes.

Reproduction commands are documented in README, INSTALLATION, DATA_GENERATION, and
EVALUATION. The final runtime requires no hosted AI service or internet connection.

## 17. Future work

Next steps require utility partnerships, licensed field telemetry, prospective incident
exercises, chemistry-specific transport validation, network-specific conformal calibration,
authenticated deployment, operator usability studies, and external scientific review.
Optional OpenVINO export, FP32/INT8 benchmarking, larger network stress tests, and sparse
attention optimization should follow only after compatible checkpoints and governed
calibration data exist.

## 18. AI assistance

ChatGPT/Codex and Claude/Claude Code were used for implementation assistance, debugging,
test generation, documentation review, and architecture critique. The project author
selected the architecture, scientific objectives, evaluation methodology, claims, and
final implementation, and validated the submitted system. AI services are not runtime
dependencies.

## 19. References

1. US EPA. Water Network Tool for Resilience (WNTR). https://usepa.github.io/WNTR/
2. US EPA. Water Quality Surveillance and Response. https://www.epa.gov/waterresilience
3. Rampasek et al. Recipe for a General, Powerful, Scalable Graph Transformer. 2022. https://arxiv.org/abs/2205.12454
4. Jaegle et al. Perceiver IO. 2021. https://arxiv.org/abs/2107.14795
5. Houlsby et al. Parameter-Efficient Transfer Learning for NLP. 2019. https://proceedings.mlr.press/v97/houlsby19a.html
6. Yu et al. Gradient Surgery for Multi-Task Learning. 2020. https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
7. Chen et al. GradNorm. 2018. https://proceedings.mlr.press/v80/chen18a.html
8. Ross et al. DAgger. 2011. https://proceedings.mlr.press/v15/ross11a.html
9. Angelopoulos and Bates. A Gentle Introduction to Conformal Prediction. 2021. https://arxiv.org/abs/2107.07511
10. Liu et al. Energy-based Out-of-distribution Detection. 2020. https://arxiv.org/abs/2010.03759
