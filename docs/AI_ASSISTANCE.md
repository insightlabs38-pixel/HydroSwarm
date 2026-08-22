# AI-assisted development disclosure

AI coding assistants were used as engineering tools during HydroSwarm development. Repository history and project records include:

- ChatGPT / Codex;
- Claude / Claude Code;
- Freebuff.

They were used for implementation assistance, debugging, test generation, documentation review, code/architecture critique, and repository maintenance.

The project author retained responsibility for scientific objectives, architecture choices, experiment governance, split/lock policy, promotion decisions, claims, safety boundaries, and final release decisions. Generated suggestions were subject to repository tests, frozen protocols, hash/provenance checks, and human review.

## Scientific boundary

The final HydroCore-v5 model is not a hosted general-purpose LLM. It is a locally loaded scientific model trained for the governed Sentinel task family on synthetic WNTR/EPANET-generated evidence. ChatGPT, Claude, and Codebuff are not inference components and have no runtime authority.

No hosted AI API is required for HydroSwarm's scientific runtime.

## Disclosure principle

AI assistance does not change the evidentiary standard applied to the repository: numerical claims must trace to generated artifacts; model/calibration identity is hash-bound; the locked evaluation was authorization-gated and opened once; and negative/stress results are reported rather than rewritten after the fact.

Generated code, tests, scientific assumptions, citations, licenses, security posture, and claims remain the project author's review responsibility.
