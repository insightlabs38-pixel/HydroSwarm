# Devpost draft

## Tagline

Offline, physics-verified decision support for drinking-water contamination incidents:
localize the source, choose the next sample, simulate response alternatives with real
hydraulic solvers, and require a human to approve before anything happens.

## Inspiration

A utility alert creates an evidence problem and a safety problem at the same time. The
source may be ambiguous, field samples are expensive, flows change, and an intuitive
response can reduce pressure or interrupt service. Operators need transparent decision
support that respects physics and uncertainty without sending sensitive infrastructure
data to a cloud service, and without ever pretending a neural estimate alone is a safe
answer.

## What it does

HydroSwarm imports an EPANET network, reconciles imperfect telemetry, localizes a
calibrated source region, recommends the next informative sample, generates diverse typed
response plans, and compares them -- including no response -- through exact WNTR
simulation. Unsafe or incomplete plans are rejected. A human must explicitly approve any
verified plan, and no infrastructure-control connector exists anywhere in the system.

The fastest way to see it: the **REFERENCE INCIDENT**, a real, checksummed, progressive
replay of a frozen WNTR-backed scenario, available from first launch with no network
import or live backend required -- alert, source uncertainty, evidence-insufficient,
sample selection, posterior contraction, plan generation, an unsafe-plan rejection, a
verified alternative, a human-approval pause, and a completed, replayable record.

## How it was built

**Scientific runtime:** Python, PyTorch, WNTR/EPANET, NetworkX, NumPy, pandas.
**Service:** FastAPI, Pydantic, SQLite, a bounded local worker, WebSocket updates.
**Console:** React, TypeScript, Vite, TanStack Query, Zustand, MapLibre GL JS, ECharts,
Cytoscape, Vitest, Playwright, and dedicated accessibility tests.
**Reproducibility and release:** safetensors, Ruff, Pyright, pytest/coverage, pip-audit,
pre-commit, GitHub Actions, Docker Buildx (published multiarch `linux/amd64`/`linux/arm64`
images).

The frozen HydroCore-v4 checkpoint (4.18M parameters) fuses with a classical
physics-derived source-signature path under a dynamic, disagreement-aware trust
coefficient, calibrated with split-conformal prediction and five-component
out-of-distribution detection.

## Challenges and accomplishments

The hardest part was making uncertainty operational end to end: disagreement changes
fusion trust; invalid calibration/OOD evidence suppresses planning; sample value is
measured as expected posterior information gain; and a neural recommendation can never
bypass deterministic constraints and exact WNTR verification.

A concrete example from the submission-readiness pass itself: the frozen model bundle's
path was originally computed relative to the source-tree layout, which silently broke for
a non-editable container install -- it would have failed closed to a degraded fallback
while still reporting healthy. Building a single shared path resolver (used by both the
API and the CLI self-test) and a build-time container gate that fails the Docker build
itself if the frozen bundle doesn't actually load turned a subtle, hard-to-notice bug into
a loud, unmissable one.

The REFERENCE INCIDENT generator enforces its own stage-correctness with internal
assertions -- it refuses to produce an artifact where, say, a plan's verification outcome
appears before the verification event that actually produced it. Getting a demo that is
both progressively revealing *and* provably never leaks future information required
building that guarantee into the generator, not just writing careful prose.

## What we learned and what is next

Scientific coherence matters more than a fluent interface, but a judge only has a few
minutes -- the REFERENCE INCIDENT exists because a fully-populated fallback screen, while
useful for resilience, is not the strongest way to show a real decision process unfold.
Next steps are field-data partnerships, utility-specific calibration, chemistry-specific
transport validation, larger network stress testing, authenticated deployment, and
prospective operator studies. Current results are simulated and do not establish field
safety or accuracy.

## Built with

Python · PyTorch · WNTR/EPANET · NetworkX · NumPy · pandas · FastAPI · Pydantic · SQLite ·
React · TypeScript · Vite · TanStack Query · Zustand · MapLibre GL JS · ECharts ·
Cytoscape · Vitest · Playwright · pytest · Ruff · Pyright · safetensors · Docker Buildx ·
GitHub Actions

## Release and submission links

- Repository: https://github.com/insightlabs38-pixel/HydroSwarm
- GitHub Release: https://github.com/insightlabs38-pixel/HydroSwarm/releases/tag/v0.1.2-hackathon
- Runtime ZIP: https://github.com/insightlabs38-pixel/HydroSwarm/releases/download/v0.1.2-hackathon/HydroSwarm-v0.1.2-hackathon-runtime.zip
- Release manifest: https://github.com/insightlabs38-pixel/HydroSwarm/releases/download/v0.1.2-hackathon/RELEASE_MANIFEST.json
- Published multiarch image: `ghcr.io/insightlabs38-pixel/hydroswarm:v0.1.2-hackathon`
- Technical report: https://github.com/insightlabs38-pixel/HydroSwarm/blob/v0.1.2-hackathon/docs/FINAL_SYSTEM.md
- Demo video: **pending final recording**
