# Quickstart

The fastest path to seeing HydroSwarm work, in order of least setup required.

## 1. Docker (no build, no dependencies to install)

```bash
docker compose -f docker-compose.release.yml up
```

Open `http://127.0.0.1:8765`. Choose **Run Reference Incident** on the first-launch
screen.

## 2. Native setup script (Linux/macOS/Windows)

```bash
git clone https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm
./setup_hydroswarm_linux.sh    # or _macos.sh, or _windows.ps1 in PowerShell
./start_hydroswarm_linux.sh    # or _macos.sh, or _windows.ps1
```

The setup script creates a project-local `.venv`, installs CPU-only dependencies,
verifies the frozen model bundle, builds the console if needed, and runs a readiness
self-test before it says done.

## What you'll see

On first launch with no incident configured, four choices:

- **Run Reference Incident** (recommended) -- a real, checksummed, progressive replay of
  the frozen golden scenario. No live backend or network import required.
- **Run Live Example** -- the real production pipeline computing now (real network
  import, real incident creation, real analysis, real WNTR verification, real approval
  pause), automatically, against known reference inputs instead of your own network or
  live telemetry. Labeled `LIVE COMPUTATION · REFERENCE INPUTS` while it runs.
- **Import Your Own Network** (advanced) -- import a real EPANET `.inp` file and start an
  incident against it yourself, through a compact form.
- **Explore illustrative fallback** -- the hand-authored, clearly-labeled demo fixture.

See [Reference demo](REFERENCE_DEMO.md) for what the reference incident actually shows,
and [Final system](FINAL_SYSTEM.md) for the exact frozen model identity behind it.

## If something doesn't work

See [Installation](INSTALLATION.md)'s troubleshooting section, or run the readiness
self-test directly:

```bash
hydroswarm self-test --human
```

It prints a checklist (Python runtime, frozen bundle, model hash, calibration,
normalization, WNTR/EPANET, SQLite, frontend assets, port availability) and says exactly
which check failed.
