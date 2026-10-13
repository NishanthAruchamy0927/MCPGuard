# syntax=docker/dockerfile:1
# MCPGuard — FastAPI inference API + research dashboard in one container.
# Works locally (docker compose up --build) and as a Hugging Face Docker Space (port 7860).

FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    MCPGUARD_LOG_JSON=1 \
    PORT=7860

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) dependencies (cached layer). CPU-only torch keeps the image small.
COPY requirements-serve.txt .
RUN pip install --retries 10 --timeout 120 torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
# xgboost's Linux wheel pulls ~300 MB of CUDA/NCCL libraries; the CPU image needs none of them.
RUN grep -v '^xgboost' requirements-serve.txt > /tmp/req.txt \
 && pip install --retries 10 --timeout 120 -r /tmp/req.txt \
 && pip install --retries 10 --timeout 120 --no-deps xgboost==3.4.1

# 2) bake the sentence-embedding model into the image (no download at start-up)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# pandas 3 string columns are pyarrow-backed, so the pickled feature pipeline needs it at load time.
RUN pip install --retries 10 --timeout 120 pyarrow==25.0.1

# 3) application code + trained artefacts (the model is never retrained at start-up)
COPY configs/ configs/
COPY src/ src/
COPY backend/ backend/
COPY frontend/ frontend/
COPY app/ app/
COPY models/ models/
COPY artifacts/metrics/ artifacts/metrics/
COPY artifacts/figures/ artifacts/figures/
COPY artifacts/experiments/ artifacts/experiments/
COPY artifacts/reports/ artifacts/reports/
COPY data/metadata/ data/metadata/
COPY data/processed/mcp_tools_clean.jsonl data/processed/splits.csv data/processed/structured_features.csv data/processed/embeddings_cache.npz data/processed/

# 4) non-root runtime user (UID 1000 matches Hugging Face Spaces)
# artifacts/db must exist (owned by appuser) so a fresh named volume mounted there inherits writable ownership.
RUN useradd --create-home --uid 1000 appuser \
 && mkdir -p /app/artifacts/db \
 && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health', timeout=4).status==200 else 1)"

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
