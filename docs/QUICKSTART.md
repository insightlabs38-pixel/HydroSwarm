# Quickstart

The fastest accurate path to the **current HydroCore-v5 source**.

## Docker: current checkout

```bash
git clone https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm
docker compose build
docker compose up
```

Open `http://127.0.0.1:8765`.

This builds the current Dockerfile, includes the frozen V5 bundle, runs the strict V5 readiness check during the build, and launches the V5-default API.

> `docker compose -f docker-compose.release.yml up` is currently pinned to the historical `v0.1.0-hackathon` image, which predates V5. Do not use that image to verify the final V5 system.

## Native

Linux:

```bash
./setup_hydroswarm_linux.sh
./start_hydroswarm_linux.sh
```

macOS and Windows have matching `_macos.sh` and `_windows.ps1` scripts.

The setup scripts finish with the strict application self-test. They currently retain a legacy V4 bundle precheck, but the final strict self-test and application runtime use V5. See [Installation](INSTALLATION.md) for that packaging caveat.

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

Some human-readable self-test labels still contain V4-era wording; the actual implementation and machine-readable trained-asset identity are V5. See [Installation](INSTALLATION.md#known-self-test-presentation-debt).

Then consult [Operator guide](USER_GUIDE.md) and [Limitations](LIMITATIONS.md).
