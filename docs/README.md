# HydroSwarm documentation map

No path here requires reading everything -- pick the one that matches why you're here.
If you are unsure which experiment/model/runtime is the actual final submission, the
answer is always [Final system](FINAL_SYSTEM.md); nothing else in this repository
overrides it.

## For judges

1. [Top-level README](../README.md) -- 60-second overview, screenshot, try-it.
2. [Quickstart](QUICKSTART.md) -- the fastest path to a running instance.
3. [Final system](FINAL_SYSTEM.md) -- exact frozen identity, hashes, measured results.
4. Run the **REFERENCE INCIDENT** from the first-launch screen after `docker compose -f
   docker-compose.release.yml up` or a native setup script -- a real, checksummed replay
   of the frozen golden scenario, no live backend required. See [Reference demo](REFERENCE_DEMO.md).
5. [Judging evidence map](JUDGING.md).

## For users

1. [Quickstart](QUICKSTART.md) / [Installation](INSTALLATION.md) -- native and container
   setup, troubleshooting.
2. [Operator guide](USER_GUIDE.md) -- the console workflow end to end.
3. [Glossary](GLOSSARY.md) -- unfamiliar term? Look here first.
4. [Limitations and failure cases](LIMITATIONS.md).

## Technical reference

- [Technical brief](TECHNICAL_BRIEF.md) -- one-page summary, links to full depth
- [Final system authority](FINAL_SYSTEM.md)
- [Full architecture](ARCHITECTURE.md)
- [Model card](MODEL_CARD.md)
- [Evaluation protocol](EVALUATION.md)
- [Dataset card](DATASET_CARD.md)
- [Data generation and governance](DATA_GENERATION.md)
- [Reference demo](REFERENCE_DEMO.md)
- [Architecture diagrams](diagrams/README.md)
- [Glossary](GLOSSARY.md)
- [Security policy](SECURITY.md)

## Research / historical work

Earlier architecture generations (HydroCore-S/M/L, a v3-era comparison superseded by the
frozen HydroCore-v4 in [Final system](FINAL_SYSTEM.md)) and exploratory experiments are
kept for research transparency, not presented as the final submission:

- [Historical benchmark section](../README.md#research-evaluation-and-historical-development)
  in the top-level README.
- [ARM migration notes](ARM_MIGRATION.md)
- [Evaluation v3 policy (superseded)](EVALUATION_V3_POLICY.md)
- `reports/results/` -- the full, dated evaluation/training report history.

## Submission-specific

- [Devpost draft](DEVPOST.md)
- [Four-minute demo script](VIDEO_SCRIPT.md)
- [Submission checklist](SUBMISSION_CHECKLIST.md)
- [AI-assistance disclosure](AI_ASSISTANCE.md)
- [References](REFERENCES.md)
