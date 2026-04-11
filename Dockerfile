# ─────────────────────────────────────────────────────────────────────────────
# XAI Learning Recommendation System — Multi-stage Docker build
# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Build the React frontend (Node.js)
# Stage 1: Install Python dependencies into /install
# Stage 2: Combine — runtime Python image with built frontend + app code
#
# Models (~90MB) are NOT baked in — downloaded from S3 at container startup.
# SQLite is replaced by PostgreSQL via DATABASE_URL env var.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 0: Frontend builder (Node.js) ──────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Install deps first (layer cached unless package.json changes)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline

# Copy source and build
# vite.config.js sets outDir: '../backend/static', but inside this stage
# that resolves to /frontend/../backend/static = /backend/static.
# We override outDir explicitly to write into /frontend/dist instead,
# then copy it to the right place in Stage 2.
COPY frontend/ .
RUN npm run build -- --outDir /frontend/dist


# ── Stage 1: Python dependency builder ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some XAI wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs (libgomp for scikit/torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY backend/ ./backend/
COPY data/fixtures/ ./data/fixtures/

# Copy the built React frontend into the FastAPI static directory
# FastAPI's StaticFiles mount and SPA catch-all route (main.py) will serve it.
COPY --from=frontend-builder /frontend/dist/ ./backend/static/

# Create directories that will be populated at runtime
RUN mkdir -p models data mlruns

# Non-root user for security
RUN addgroup --system xai && adduser --system --ingroup xai xaiuser
RUN chown -R xaiuser:xai /app
USER xaiuser

EXPOSE 8000

# Health check — hits /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startup: S3 model download happens inside lifespan() before serving requests
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

