# HydroSwarm documentation map

The final HydroSwarm submission serves the frozen **HydroCore-v5** system. Start with the path that matches your review goal; historical V3/V4/M9/M10 material is preserved but never overrides [Final system](FINAL_SYSTEM.md).

## New reader / domain expert path

[Executive summary](EXECUTIVE_SUMMARY.md) → [Scientific evidence](SCIENTIFIC_EVIDENCE.md) → [Authority and safety](AUTHORITY_AND_SAFETY.md)

Read this first if you are new to the project or reviewing it from a water-infrastructure or general technical background. It explains the problem, the system, the final results, and the limitations without requiring any other document.

## Judge skim path

[Top-level README](../README.md) → [Scientific evidence](SCIENTIFIC_EVIDENCE.md) → [Judging evidence map](JUDGING.md)

This path answers: what is the system, what did the one-time final test measure, and where is the proof?

## Technical reviewer path

[Technical brief](TECHNICAL_BRIEF.md) → [Final system](FINAL_SYSTEM.md) → [Architecture](ARCHITECTURE.md) → [Authority and safety](AUTHORITY_AND_SAFETY.md) → [Scientific evidence](SCIENTIFIC_EVIDENCE.md)

Use this path to verify serving identity, learned/runtime/authority boundaries, failure behavior, and final evidence.

## Scientific reviewer path

[Model card](MODEL_CARD.md) → [Evaluation](EVALUATION.md) → [Dataset card](DATASET_CARD.md) → [Reproducibility](REPRODUCIBILITY.md) → [Claims and evidence](CLAIMS_AND_EVIDENCE.md)

Use this path to audit supervision scope, split governance, final lock, synthetic-data limits, and exact claim wording.

## Operator path

[Quickstart](QUICKSTART.md) → [Operator guide](USER_GUIDE.md) → [Limitations](LIMITATIONS.md)

For V5, run the published `docker-compose.release.yml` image (`ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.1`) or build/run the current source; both serve the same frozen V5 identity.

## Core current documentation

- [Executive summary](EXECUTIVE_SUMMARY.md)
- [Final system authority](FINAL_SYSTEM.md)
- [Scientific evidence dossier](SCIENTIFIC_EVIDENCE.md)
- [Authority and safety](AUTHORITY_AND_SAFETY.md)
- [Architecture](ARCHITECTURE.md)
- [Model card](MODEL_CARD.md)
- [Evaluation protocol and final evidence](EVALUATION.md)
- [Dataset card](DATASET_CARD.md)
- [Data generation and governance](DATA_GENERATION.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Claims and evidence ledger](CLAIMS_AND_EVIDENCE.md)
- [Technical brief](TECHNICAL_BRIEF.md)
- [Installation](INSTALLATION.md)
- [Quickstart](QUICKSTART.md)
- [Operator guide](USER_GUIDE.md)
- [Limitations](LIMITATIONS.md)
- [Glossary](GLOSSARY.md)
- [Judging evidence map](JUDGING.md)
- [References](REFERENCES.md)
- [AI-assistance disclosure](AI_ASSISTANCE.md)
- [Architecture diagrams](diagrams/README.md)
- [Security policy](SECURITY.md)

## Demo / product documentation

- [Reference incident](REFERENCE_DEMO.md) — workflow demonstration, not final V5 benchmark evidence.
- [Problem and product boundary](PROBLEM.md)
- [Devpost submission copy](DEVPOST.md)
- [Final video narration/script](VIDEO_SCRIPT.md)
- [Video and caption materials](video/)

The submission artifacts above mirror the frozen v0.2.1 submission story and are not scientific authorities.

## Historical research

Historical records remain available for transparency:

### V4 / capability-remediation era

- `reports/results/v4/`
- `docs/evaluation/`
- historical V4 release bundle under `models/hydrocore-v4-release/`
- [V4 architecture diagram](diagrams/hydrocore-v4.mmd)

Statements in those artifacts that the V5 lock was not yet opened are historical in time; current V5 lock status is M11.6 PASS.

### V5 development milestones before final lock

- `reports/evaluation/hydrocore-v5/m9*` — architecture/training/capacity search
- `reports/evaluation/hydrocore-v5/m10/` — downstream authority/full trajectories/serving freeze
- `reports/evaluation/hydrocore-v5/m11/` — finalist freeze, full validation, one-time final lock

M9/M10 are current V5 research lineage but **development evidence**, not the final test.

### Earlier S/M/L generation

- `reports/results/medium-evaluation-final.json`
- `reports/results/topology-transfer-m.json`
- [Evaluation historical section](EVALUATION.md#historical-record)

These results are not current V5 claims.

## One rule for conflicts

If current prose conflicts with immutable artifacts, the generated frozen artifacts win. Open a documentation defect rather than averaging, guessing, or silently choosing the more favorable number.
