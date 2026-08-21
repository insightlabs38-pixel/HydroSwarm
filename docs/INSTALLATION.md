# Installation and offline launch

> **Current model identity:** the source application serves the frozen HydroCore-v5 bundle through `V5PipelineFactory(resolve_v5_bundle_dir())`. The historical `docker-compose.release.yml` image predates V5; use one of the V5 source paths below for the final system.

## Runtime boundary

```mermaid
flowchart TD
  B["Browser"] <-->|"localhost:8765"| F["FastAPI + built console"]
  F <--> D["SQLite / local files"]
  F <--> V["HydroCore-v5 + classical pipeline"]
  V <--> W["WNTR / EPANET"]
  X["No hosted model API"]
  Y["No required cloud runtime"]
  Z["No autonomous actuation connector"]
```

Current diagram source: [diagrams/offline-deployment-v5.mmd](diagrams/offline-deployment-v5.mmd).

## Requirements

- 64-bit Python 3.12+
- Node.js 22+ only when the frontend must be rebuilt
- roughly 4 GiB RAM for the small/default local demonstration
- Docker, if using the container path
- installation-time internet access to obtain dependencies unless they are already cached
- runtime internet access is not required by the scientific pipeline

## V5 path A: build the current Docker checkout

This is the cleanest container path to the **current source**:

```bash
git clone https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm
docker compose build
docker compose up
```

Then open `http://127.0.0.1:8765`.

The current `Dockerfile`:

- copies the frozen `models/hydrocore-v5-release/` bundle;
- sets `HYDROSWARM_V5_BUNDLE_DIR`;
- builds the frontend;
- includes the reference-demo/frozen runtime fixtures;
- runs `run_self_test(strict=True)` during image construction;
- runs the API that defaults to V5.

The developer compose file publishes `127.0.0.1:8765`, drops Linux capabilities, prevents privilege escalation, uses a read-only root filesystem, and persists `/data`.

## Important: historical release-compose image

```bash
docker compose -f docker-compose.release.yml up
```

is **not the final V5 path at the current repository state**. That file is still pinned to `ghcr.io/insightlabs38-pixel/hydroswarm:v0.1.0-hackathon`, an image published before the V5 packaging rebase; no new image has been published under that tag yet. `docker-compose.release.yml` will be repointed at a new immutable V5 image tag/digest once one is published.

This is a repository packaging/versioning follow-up, not a model/evaluation ambiguity. Do not use the historical release image to verify V5 identity; use [V5 path A](#v5-path-a-build-the-current-docker-checkout) (developer compose, built from this checkout's own current Dockerfile) or [V5 path B](#v5-path-b-native-setup-and-launch) below instead.

## V5 path B: native setup and launch

```bash
git clone https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm

./setup_hydroswarm_linux.sh
./start_hydroswarm_linux.sh
```

macOS (Apple Silicon / arm64):

```bash
./setup_hydroswarm_macos.sh
./start_hydroswarm_macos.sh
```

Native macOS support targets Apple Silicon (arm64) only. The frozen runtime requires `torch>=2.5`, for which official macOS Intel/x86_64 binary support is unavailable upstream; `setup_hydroswarm_macos.sh` fails early with this explanation on Intel Macs rather than failing obscurely mid-install.

Windows PowerShell:

```powershell
.\setup_hydroswarm_windows.ps1
.\start_hydroswarm_windows.ps1
```

The platform setup scripts create a project-local `.venv`, install dependencies, build the frontend when needed, and finish with the strict application self-test.


## Manual native install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cd frontend
npm ci
npm run build
cd ..
hydroswarm self-test --strict --human
hydroswarm start
```

Windows uses `.venv\Scripts\python`, `npm.cmd`, and the PowerShell launcher as appropriate.

## Verify the final V5 identity

Run:

```bash
hydroswarm self-test --strict
```

The machine-readable report includes `trained_assets.architecture_version`, `trained_assets.model_sha256`, `trained_assets.bundle_dir`, calibration status, bounded inference/simulation checks, resources, frontend assets, and the reference artifact.

For the final frozen system, the model hash must be:

```text
de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5
```

The release manifest itself is [models/hydrocore-v5-release/runtime_manifest.json](../models/hydrocore-v5-release/runtime_manifest.json), and the authoritative freeze is [FINAL_SYSTEM.md](FINAL_SYSTEM.md).

## Runtime behavior if V5 assets are invalid

The V5 loader verifies:

- release schema;
- five-output runtime allowlist;
- `sentinel` trained-task allowlist;
- feature-schema identity;
- fusion identity;
- release file hashes;
- checkpoint hash;
- calibration artifact identity.

A failure makes the trained V5 assets unavailable and the pipeline fails closed toward classical-safe behavior. It never silently falls back to V4.

## Native Windows versus Linux/Docker

HydroSwarm's exact simulator runs in a killable subprocess with a hard timeout. Linux/macOS can use `fork`; Windows uses `spawn`, which starts a new Python interpreter and re-imports scientific dependencies. As a result, native Windows simulator-heavy workloads have materially higher process-start overhead.

For production-equivalent demo latency and the most exhaustive real-simulator test path on a Windows host, prefer Docker Desktop/WSL2.

## Offline operation

The scientific runtime does not require a hosted model service. The application is designed for local operation with local model, calibration, network, database, and reference assets. Dependency downloads and container/image acquisition are installation concerns, not runtime inference dependencies.

## Troubleshooting

- Port occupied: pass `--port` to `hydroswarm start`.
- Strict self-test says V5 assets unavailable: inspect the reported `bundle_dir`, fallback reason, and checkpoint hash.
- Frontend missing: rebuild `frontend/dist`.
- WNTR/EPANET check fails: verify the installed simulator dependency/runtime.
- Do not delete the incident database to “fix” a migration or evidence inconsistency; preserve the record and inspect the error.
- Do not use the historical release-compose image when validating V5.

## Reproducibility checks

Useful non-locked checks include:

```bash
hydroswarm self-test --strict
python -m pytest
python -m pyright
python -m ruff check src tests scripts
```

These commands do not authorize rerunning the final locked M11.6 evaluation. See [Reproducibility](REPRODUCIBILITY.md).
