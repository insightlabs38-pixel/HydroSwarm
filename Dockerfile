FROM node:22-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HYDROSWARM_DATA_DIR=/data
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
RUN mkdir -p /data && chown -R hydroswarm:hydroswarm /app /data
USER hydroswarm
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)"
CMD ["hydroswarm", "start", "--host", "0.0.0.0", "--port", "8765", "--allow-network-bind"]
