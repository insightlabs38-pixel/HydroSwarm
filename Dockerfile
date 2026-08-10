FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HYDROSWARM_DATA_DIR=/data \
    HYDROSWARM_V4_BUNDLE_DIR=/app/models/hydrocore-v4-release \
    HYDROSWARM_REFERENCE_DEMO_PATH=/app/artifacts/reference-demo/reference-incident-v1.json \
    HYDROSWARM_FROZEN_SCENARIO_DIR=/app/data/frozen \
    TMPDIR=/tmp
WORKDIR /app
RUN useradd --create-home --uid 10001 hydroswarm
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
# wntr publishes prebuilt wheels for macOS and Linux x86_64
# (manylinux2014_x86_64) but NOT Linux ARM64 -- confirmed against PyPI's
# own file listing for the pinned version, no aarch64 wheel exists for any
# supported Python version. On linux/arm64 (SUB-12.1 #23's Docker CI gate
# build target) pip therefore falls back to building wntr from source,
# which needs a C++ compiler the slim base image does not ship by
# default (a real linux/arm64 build failed on exactly this: "error:
# [Errno 2] No such file or directory: 'g++'"). Installed and purged in
# the same layer so the final image does not carry a compiler toolchain
# it never needs at runtime; on linux/amd64 this is a harmless no-op
# since a wheel is already available there and nothing gets compiled.
#
# Force the CPU wheel in the local decision-support image; the default Linux PyPI
# resolution can pull multi-gigabyte CUDA components that this runtime never uses.
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" \
 && python -m pip install --no-cache-dir . \
 && apt-get purge -y --auto-remove g++ \
 && rm -rf /var/lib/apt/lists/*
COPY --from=frontend /build/frontend/dist frontend/dist
COPY configs/ configs/
# Submission-readiness SUB-1 (P0): the frozen, self-contained V4 inference
# release bundle must be baked into the image and served from the path
# HYDROSWARM_V4_BUNDLE_DIR points at above -- this app is a *non-editable*
# `pip install .`, so the pre-existing source-tree-relative path inference
# (Path(__file__).resolve().parents[...]) does not land on /app from inside
# site-packages. Without this COPY+ENV pair the container would silently
# fail closed to the classical-safe fallback while still reporting healthy.
COPY models/hydrocore-v4-release/ models/hydrocore-v4-release/
# SUB-4/SUB-3: bake in the governed REFERENCE INCIDENT artifact so the
# judge demo path works fully offline in the container, served at
# HYDROSWARM_REFERENCE_DEMO_PATH above via GET /api/reference-demo.
COPY artifacts/reference-demo/ artifacts/reference-demo/
# SUB-12.1 P1 #4: the frozen golden network/scenario fixture the LIVE
# example's real reference inputs (GET /api/live-example-inputs) are
# computed from -- without this the LIVE example judge path 404s inside
# the container even though it works from a source checkout.
COPY data/frozen/ data/frozen/
# The real LIVE pipeline resolves pristine topology fixtures while building
# classical signature priors.  These are runtime inputs, not training data;
# without them a hardened container reaches the live analysis path and then
# fails closed with a missing relative topology file.
COPY data/topologies/ data/topologies/
# Governed classical signature input required by the real frozen LIVE
# pipeline; copy this one manifest, never the learning corpus/checkpoints.
COPY data/learning-v2/cycle-b2/signatures/loop-grid.json data/learning-v2/cycle-b2/signatures/loop-grid.json
RUN mkdir -p /data && chown -R hydroswarm:hydroswarm /app /data
USER hydroswarm
# Container self-test gate: fails the build (not just a post-hoc CI check)
# if the frozen bundle baked in above does not actually load, hash-verify,
# and report ready inside this exact image; if calibration is not FITTED;
# if the frontend was not built in; or if the reference-demo artifact is
# missing (SUB-12.1 #21: the same --strict gate the native setup scripts,
# CI, and the release workflow use) -- catching a broken release image
# before it is ever pushed.
RUN python -c "\
import json, sys; \
from hydroswarm.cli import run_self_test; \
result = run_self_test(strict=True); \
print(json.dumps(result, indent=2, sort_keys=True)); \
sys.exit(0 if result['ok'] else 1)"
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)"
CMD ["hydroswarm", "start", "--host", "0.0.0.0", "--port", "8765", "--allow-network-bind"]
