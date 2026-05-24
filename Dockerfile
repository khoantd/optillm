# Build stage
FROM python:3.12-slim-bookworm AS builder

ARG PORT=8000
ENV OPTILLM_PORT=$PORT

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .

# CPU-only PyTorch is much smaller than the default CUDA-enabled wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.docker.txt

RUN find /usr/local/lib/python3.12/site-packages -type d -name tests -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.12/site-packages -name '*.py[co]' -delete 2>/dev/null || true

# Final stage
FROM python:3.12-slim-bookworm

ARG PORT=8000
ENV OPTILLM_PORT=$PORT

LABEL org.opencontainers.image.source="https://github.com/codelion/optillm"
LABEL org.opencontainers.image.description="OptiLLM full image with model serving and API routing capabilities"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY optillm/ optillm/
COPY optillm.py README.md ./

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE ${PORT}

ENTRYPOINT ["python", "optillm.py"]
