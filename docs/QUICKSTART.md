# Quickstart

The fastest accurate path to the **current HydroCore-v5 source**.

## Docker: published release (recommended)

```bash
git clone https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm
docker compose -f docker-compose.release.yml up
```

Open `http://127.0.0.1:8765`.

This pulls the published, tested multiarch `v0.2.1` image (`ghcr.io/insightlabs38-pixel/hydroswarm:v0.2.1`) directly rather than building locally.

## Docker: build from this checkout

```bash
docker compose build
docker compose up
```

Open `http://127.0.0.1:8765`.

This builds the current Dockerfile, includes the frozen V5 bundle, runs the strict V5 readiness check during the build, and launches the V5-default API. Useful when reviewing an uncommitted or in-progress change rather than the tagged release.

## Native

Linux:

```bash
./setup_hydroswarm_linux.sh
./start_hydroswarm_linux.sh
```

macOS and Windows have matching `_macos.sh` and `_windows.ps1` scripts. Native macOS support is Apple Silicon/ARM64 only; native macOS Intel/x86_64 is not supported (no upstream `torch>=2.5` wheel exists for it). Native Windows support is x86_64.

The setup scripts finish with the strict application self-test, verifying the frozen V5 release bundle throughout.

## Verify the model that will serve

```bash
hydroswarm self-test --strict
```

The frozen V5 model SHA-256 is:

`de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5`

Exact identity: [Final system](FINAL_SYSTEM.md).

## What to open first

On first launch:

- **Run Reference Incident** — deterministic checksummed replay of the frozen golden workflow. It demonstrates workflow/authority, not final V5 benchmark performance.
- **Run Live Example** — executes the real API pipeline against known reference inputs.
- **Import Your Own Network** — advanced path for a local EPANET `.inp`.
- **Illustrative fallback** — clearly labeled hand-authored fallback.

For final V5 evidence, do not infer performance from the demo. Read [Scientific evidence](SCIENTIFIC_EVIDENCE.md).

## How to read a result

1. Check provenance/model hash.
2. Check calibration applicability and OOD state.
3. Treat the learned source/event/evidence outputs as advisory.
4. Treat Scout/planning decisions as deterministic governed controls.
5. Require exact WNTR/EPANET verification for `VERIFIED`.
6. Require a separate human event for `APPROVED`.
7. Remember that approval records a decision; HydroSwarm does not actuate infrastructure.

## Problems?

Run:

```bash
hydroswarm self-test --human --strict
```

Then consult [Operator guide](USER_GUIDE.md) and [Limitations](LIMITATIONS.md).
