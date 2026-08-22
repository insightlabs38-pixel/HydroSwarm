# Limitations and failure cases

HydroSwarm is research decision support. It has not been validated for live utility operation, public-health advisories, regulatory decisions, or autonomous infrastructure control.

## Scientific limitations

### Synthetic evidence only

All current model-development and final locked evidence is WNTR/EPANET-generated. The final program spans multiple synthetic topology families and stress conditions, but simulator diversity does not establish field generalization.

### Nominal and stress performance differ

The locked nominal slice (`n=15`) reached 73.3% Top-1 and 86.7% Top-3. Across all seven locked-final conditions (`n=105`), Top-1 fell to 55.2% and Top-3 to 76.2%.

That difference matters. The final system should not be described using nominal performance alone.

### Condition-specific weak points exist

The 15-case locked sensor-dropout slice reached 66.7% applicable conformal coverage, below the 85% aggregate locked-final gate threshold. Ambiguity/disagreement and measurement noise also produced lower Top-1 and actionability than nominal.

The aggregate gate passed; individual weak slices remain limitations rather than being hidden or used to redefine the gate after the fact.

### Novel topology is not calibrated

The locked novel-topology population (`n=20`) had:

- Top-1 55%;
- Top-3 70%;
- MRR 0.652;
- `calibrated_rate=0`;
- actionable rate 0%;
- human-approved rate 0%.

The predictive metrics are descriptive/non-gating. The result demonstrates fail-closed authority, **not calibrated novel-topology capability**.

### Conformal coverage is marginal

Split-conformal coverage is a population-level marginal property under applicable conditions. It is not a per-incident confidence statement and cannot guarantee the true source is in one particular candidate set.

### Hydraulic-model dependence

WNTR/EPANET outcomes inherit:

- network topology/model errors;
- demand uncertainty;
- pump/valve/control state error;
- tank and mixing assumptions;
- sensor time/unit errors;
- water-quality transport assumptions.

A completed simulation can verify a plan against the **modeled** constraints and still be wrong about reality if the model/state is wrong.

### Train/serve feature-semantics deviation

The selected M9.6 training record used a fixed unobserved-age sentinel. The M10.4-tested serving behavior retained an `incident_elapsed` sentinel. This deviation was frozen and disclosed; it was not corrected after the final lock.

## Model scope limits

HydroCore-v5's valid trained task family is `sentinel`. Five learned outputs are runtime-enabled. Learned OOD, Scout, Strategist, `next_step`, and consequence-control outputs are not authoritative final capabilities even though relevant head structures exist in the architecture.

The model does not identify:

- contaminant chemistry;
- toxicity;
- pathogen viability;
- regulatory compliance;
- laboratory confirmation requirements.

## Sampling limitations

Deterministic Scout recommends evidence locations under modeled accessibility, delay/cost, and information criteria. It cannot know unmodeled field access, staffing, safety, chain-of-custody, lab turnaround, or sample contamination.

A recommendation is evidence-acquisition advice, not an autonomous dispatch.

## Planning limitations

Deterministic planning explores bounded typed candidates. The available action vocabulary and hard simulator budget cannot enumerate every response a utility engineer might consider. Absence of a safe generated plan does not imply no safe real-world response exists.

A `VERIFIED` plan only means the configured WNTR/EPANET run completed and passed software constraints. It does not certify real-world safety.

## Human-approval boundary

Human approval is necessary but not sufficient for real-world safety. HydroSwarm's approval event records that a person accepted a modeled verified plan. It does not validate operator credentials, utility procedures, regulation, incident-command authorization, or current field conditions.

HydroSwarm contains no autonomous actuation connector.

## Operational/runtime limits

- Local API design is not an authenticated internet-facing multi-tenant service.
- Imported network quality can be insufficient despite syntactic validity.
- Exact simulation can time out or fail numerically; those plans fail closed.
- Native Windows has higher simulator subprocess overhead than Linux/Docker.
- `docker-compose.release.yml` points to the published `v0.2.1` V5 image; building current source (`docker compose build && docker compose up`) is an equally valid, current V5 path.

## Final-evaluation limitations

The M11.6 final set was deliberately opened once. The sample sizes—105 applicable locked-final incidents and 20 novel-topology incidents—support the reported measured rates for those generated populations; they do not exhaust the space of water networks, failures, response actions, or adversarial conditions.

The lock should not be rerun merely to reduce uncertainty after observing results; doing so would weaken the held-out governance claim.

## What the zero safety counters mean

All 15 frozen hard safety counters were zero in M11.6. This is strong evidence that the tested software authority invariants held on all 125 locked incidents. It is **not** a guarantee that no unsafe behavior can occur outside those tested paths or that a real-world action is safe.

See [Authority and safety](AUTHORITY_AND_SAFETY.md) and [Scientific evidence](SCIENTIFIC_EVIDENCE.md).
