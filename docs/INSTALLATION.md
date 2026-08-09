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

## Performance: native Windows vs. Linux/Docker

HydroSwarm's optimized, production-equivalent runtime target is **Linux**
(Docker, amd64/arm64) -- see [Container](#container) above. Native Windows is a
fully supported, correct install path (the complete backend test suite, including
every real WNTR/EPANET scientific test, passes there), but it has materially
higher latency for exact hydraulic/water-quality simulation specifically, for a
real, unavoidable platform reason:

HydroSwarm's simulator wrapper runs every real WNTR/EPANET call in a genuine,
killable OS subprocess with a hard wall-clock timeout (not a daemon thread, which
Python cannot forcibly stop). On Linux/macOS this uses `multiprocessing`'s `"fork"`
start method -- a near-instant, sub-millisecond way to hand a live incident
analysis off to a child process. Windows has no `fork()` syscall at all, so the
same code correctly falls back to `"spawn"` there -- the only start method Windows
supports -- which starts a brand-new Python interpreter and re-imports NumPy,
pandas, WNTR, and Torch for every single exact simulation call. That is a real,
multi-second-per-call cost on native Windows that does not exist on Linux/macOS.

This does not affect correctness: HydroSwarm's Windows CI matrix leg runs the full
backend suite (a broad correctness/portability pass plus a dedicated real-simulator
smoke group proving native WNTR/EPANET/timeout/exception-propagation/plan-verification
behavior through the real "spawn" path) and requires it to pass. It only affects
per-simulation latency for interactive/repeated exact verification.

**If you are running HydroSwarm natively on Windows and need production-equivalent
exact-simulation latency, use Docker Desktop (WSL2 backend) and the Container path
above instead of the native PowerShell install** -- that runs the real Linux
container image, restoring the `"fork"` path. Do not expect native Windows
performance parity with Linux for workloads that make many exact simulator calls
(e.g. bulk plan comparison, corpus generation, training); native Windows remains
appropriate for a demo/self-test/light interactive session.

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
