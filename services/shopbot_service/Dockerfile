# ═══════════════════════════════════════════════════════════════════
# ShopBot Service — Multi-stage Dockerfile
# ═══════════════════════════════════════════════════════════════════
# Stage 1: Builder — installs all dependencies
# Stage 2: Runtime — minimal image for production
#
# Build: docker build -t shopfeed/shopbot:latest .
# Run:   docker run -p 8070:8070 --env-file .env shopfeed/shopbot:latest
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────── STAGE 1: Builder ────────────────────────
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# Copy requirements first for Docker layer cache optimization
COPY services/shopbot_service/requirements.txt ./requirements.txt

# Use pip with cache mount (BuildKit) for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# ─────────────────────── STAGE 2: Runtime ────────────────────────
FROM python:3.11-slim AS runtime

# Install minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security (2026 best practice)
RUN groupadd -r shopbot && useradd -r -g shopbot -s /bin/false shopbot

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
# Only copy what we need — keeps image small
COPY services/shopbot_service/ ./services/shopbot_service/
COPY shared/ ./shared/ 2>/dev/null || true

# Pre-download embedding model at build time
# This bakes the model into the image → instant cold start
# Comment out if you prefer to download at runtime (saves image space)
ARG DOWNLOAD_MODEL=true
RUN if [ "$DOWNLOAD_MODEL" = "true" ]; then \
    python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('intfloat/multilingual-e5-large-instruct')"; \
    fi

# Set ownership
RUN chown -R shopbot:shopbot /app
USER shopbot

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    SHOPBOT_API_HOST=0.0.0.0 \
    SHOPBOT_API_PORT=8070 \
    SHOPBOT_ENVIRONMENT=production \
    SHOPBOT_LOG_LEVEL=INFO

EXPOSE 8070

# Health check for Docker / Kubernetes
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8070/api/v1/shopbot/health || exit 1

# Start with uvicorn (use gunicorn in production for process management)
CMD ["uvicorn", \
     "services.shopbot_service.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8070", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info", \
     "--access-log"]
