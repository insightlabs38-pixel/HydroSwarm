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
    HYDROSWARM_FROZEN_SCENARIO_DIR=/app/data/frozen
WORKDIR /app
RUN useradd --create-home --uid 10001 hydroswarm
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
# Force the CPU wheel in the local decision-support image; the default Linux PyPI
# resolution can pull multi-gigabyte CUDA components that this runtime never uses.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" \
 && python -m pip install --no-cache-dir .
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
RUN mkdir -p /data && chown -R hydroswarm:hydroswarm /app /data
USER hydroswarm
# Container self-test gate: fails the build (not just a post-hoc CI check)
# if the frozen bundle baked in above does not actually load, hash-verify,
# and report ready inside this exact image, or if the frontend was not
# built in -- catching a broken release image before it is ever pushed.
RUN python -c "\
import json, sys; \
from hydroswarm.cli import run_self_test; \
result = run_self_test(); \
print(json.dumps(result, indent=2, sort_keys=True)); \
sys.exit(0 if result['trained_assets']['ready'] and result['frontend_assets'] == 'built' else 1)"
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)"
CMD ["hydroswarm", "start", "--host", "0.0.0.0", "--port", "8765", "--allow-network-bind"]
