# Devpost draft

## Tagline

Offline multi-agent intelligence that localizes water-quality incidents, selects evidence,
and verifies response plans with hydraulic simulation.

## Inspiration

A utility alert creates an evidence problem and a safety problem at the same time. The
source may be ambiguous, field samples are expensive, flows change, and an intuitive
response can reduce pressure or interrupt service. Operators need transparent decision
support that respects physics and uncertainty without sending sensitive infrastructure
data to a cloud service.

## What it does

HydroSwarm imports an EPANET network, reconciles imperfect telemetry, localizes a calibrated
source region, recommends the next informative sample, generates diverse typed response
plans, and compares them—including no response—through exact WNTR simulation. Unsafe or
incomplete plans are rejected. A human must explicitly approve any verified plan, and no
control connector exists.

## How it was built

Python, PyTorch, WNTR/EPANET, NetworkX, FastAPI, Pydantic, SQLite, React, TypeScript, Vite,
TanStack Query, Zustand, MapLibre GL JS, Plotly/ECharts, Cytoscape, Vitest, Playwright,
pytest, Ruff, Pyright, uv, and Docker Compose.

## Challenges and accomplishments

The hardest part was making uncertainty operational: disagreement changes fusion trust;
invalid calibration/OOD evidence suppresses planning; sample value is measured as expected
posterior information gain; and a neural recommendation can never bypass deterministic
constraints and WNTR. The frozen golden scenario and benchmark suite generate auditable
outputs rather than relying on a static demo fixture.

## What we learned and what is next

Scientific coherence matters more than a fluent interface. Next steps are field-data
partnerships, utility-specific calibration, chemistry-specific transport validation,
larger network stress testing, authenticated deployment, and prospective operator studies.
Current results are simulated and do not establish field safety or accuracy.

## Links to complete before submission

- Repository: https://github.com/insightlabs38-pixel/HydroSwarm
- Demo video: **pending final recording**
- Technical report: **pending generated release URL**
