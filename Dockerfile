# Dockerfile

# Builds the DocuMind production image.
#
# Layer strategy (ordered from least to most frequently changed):
#   1. Base OS + system packages   — almost never changes
#   2. Python dependencies         — changes only when requirements.txt changes
#   3. ML model weights            — changes only when model names change
#   4. Application source code     — changes on every commit
#
# This ordering maximises Docker layer cache hits during development.
# When you only change Python files, Docker reuses layers 1–3 (~5 min saved).

# Build:
#   docker build -t documind .
#
# Run:
#   docker run --env-file .env -p 8000:8000 documind


FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# gcc/g++ are required to compile some pip packages (rank-bm25, tokenizers).
# curl is used for Docker health checks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy only requirements.txt first so this layer is cached independently
# of your source code. A pip install only reruns when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Pre-download ML model weights ─────────────────────────────────────────────
# Baking model weights into the image means:
#   - No internet access needed at container startup
#   - Zero model download latency on Railway / cold starts
#   - Consistent model versions across all deployments
#
# all-MiniLM-L6-v2:                ~90 MB  (embedding model)
# cross-encoder/ms-marco-MiniLM-L-6-v2: ~85 MB  (reranker)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Downloading reranker model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('All models downloaded.') \
"

# ── Application source code ────────────────────────────────────────────────────
# Copied last — this is the layer that changes on every commit.
# Because it comes after the model download layer, model weights stay cached.
COPY . .

# ── Runtime directories ────────────────────────────────────────────────────────
RUN mkdir -p \
        data/documents \
        data/chroma_db \
        evaluation/results \
        logs

# ── Port ──────────────────────────────────────────────────────────────────────
# Railway injects $PORT dynamically. We expose 8000 as the default for
# local Docker use. start.py reads $PORT at runtime.
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────────
# Docker will call GET /health every 30s.
# start-period gives the container 5 minutes to run ingest before failing.
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=300s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# ── Entry point ────────────────────────────────────────────────────────────────
# start.py checks for built indexes, runs ingest.py if missing, then
# launches uvicorn. See start.py for full startup logic.
CMD ["python", "start.py"]