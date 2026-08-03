# Installation and offline launch

## Requirements

- Python 3.12 (64-bit)
- Node.js 22 only when rebuilding the frontend
- Approximately 4 GiB RAM for the small/default demonstration; more for large models
- WNTR's EPANET runtime, installed as a Python dependency

## Native install

```powershell
git clone https://github.com/insightlabs38-pixel/HydroSwarm.git
cd HydroSwarm
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
cd frontend
npm.cmd ci
npm.cmd run build
cd ..
.venv\Scripts\hydroswarm self-test
.\start_hydroswarm.bat
```

Linux/macOS equivalents use `.venv/bin/python`, `npm ci && npm run build`, and
`./start_hydroswarm.sh`. The app opens at `http://127.0.0.1:8765`. Dependency downloads
are required only during installation; runtime operation is offline.

## Container

```text
docker compose build
docker compose up
```

Compose publishes only `127.0.0.1:8765`, removes Linux capabilities, prevents privilege
escalation, uses a read-only root filesystem, and persists application state in the
`hydroswarm-data` volume.

## Reproducibility checks

```powershell
hydroswarm self-test
python -m pytest --cov=hydroswarm --cov-branch
python -m ruff check src tests
python -m pyright
python -m build --wheel --no-isolation
cd frontend
npm.cmd run lint
npm.cmd run test -- --run
npm.cmd run build
```

`self-test` executes real model inference and a bounded WNTR simulation, checks SQLite,
resource availability, bind-port status, dependency versions, and frontend assets, and
emits machine-readable hashes. A failure exits nonzero.

## Troubleshooting

- If port 8765 is occupied, pass `--port` to `hydroswarm start`.
- If memory is constrained, use a small checkpoint or classical-safe mode.
- A missing frontend build does not invalidate scientific APIs, but self-test reports
  `source-only`; run the frontend build before a demo.
- Delete no incident database to “fix” a migration. Preserve it and inspect the startup
  error; production-like evidence must remain recoverable.
