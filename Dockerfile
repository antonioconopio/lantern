# Lantern: document Q&A service (FastAPI + LangChain)

# ---- Build stage: install dependencies into a virtualenv ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build tools for any dependency that needs to compile native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only PyTorch for local embeddings; the default Linux wheel bundles
# CUDA and adds several GB to the image
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install -r requirements.txt


# ---- Runtime stage: slim image with only what's needed to run ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHROMA_PERSIST_DIR=/data/chroma \
    UPLOAD_DIR=/data/uploads

# Run as a non-root user
RUN useradd --create-home --uid 1000 lantern

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app/ ./app/
COPY lantern/ ./lantern/

# Persistent storage for the vector store and uploaded files
RUN mkdir -p /data/chroma /data/uploads && chown -R lantern:lantern /data
VOLUME ["/data"]

USER lantern

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]