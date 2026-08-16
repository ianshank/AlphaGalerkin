# AlphaGalerkin Multi-Stage Dockerfile (v0.4.0)
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]"

# Copy application source
COPY src/ src/
COPY config/ config/
COPY tests/ tests/
COPY docs/ docs/
COPY Makefile .
COPY README.md .

# Run as non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command: run fast verification tests
CMD ["pytest", "tests/sanity", "tests/security", "tests/benchmarks", "tests/regression", "-v"]
