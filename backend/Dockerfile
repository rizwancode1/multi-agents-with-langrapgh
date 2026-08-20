# Backend: FastAPI + LangGraph multi-agent API
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production

WORKDIR /app

# Install build essentials required by some Python deps (chromadb, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install project and dependencies first (better layer caching)
COPY pyproject.toml README.md ./
COPY app ./app
COPY main.py ./
RUN pip install .

# Runtime data directories (persisted via volumes in compose)
RUN mkdir -p /data
ENV CHECKPOINT_PATH=/data/checkpoints.db

EXPOSE 8000

CMD ["python", "main.py"]