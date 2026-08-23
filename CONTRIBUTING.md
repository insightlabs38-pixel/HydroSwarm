# Contributing

HydroSwarm is safety-sensitive research software. Changes must preserve the product
boundary: local decision support only, physics before fluency, exact verification before a
plan can be labeled `VERIFIED`, and explicit operator approval before a verified response can
become `APPROVED`.

Create focused commits, add regression tests for every scientific or safety behavior, and
run:

```powershell
uv sync --all-extras --dev
uv run ruff check src tests
uv run pyright
uv run pytest --cov=hydroswarm
uv run hydroswarm self-test
```

Never commit utility network data, incident telemetry, credentials, generated checkpoints,
or unlicensed datasets. Scientific claims require a versioned evaluation manifest,
repeated seeds, uncertainty intervals, and a reproducible report artifact.

