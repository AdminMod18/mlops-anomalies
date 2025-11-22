# =============================================================================
# MLOps Anomaly Detection - Production Dockerfile
# =============================================================================

# Stage 1: Builder
FROM python:3.10-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim as production

LABEL maintainer="MLOps Team" \
      version="1.0.0" \
      description="MLOps Anomaly Detection API"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    APP_USER=appuser \
    APP_PORT=8000 \
    MODEL_PATH=/app/models/isolation_forest_model.joblib \
    SCALER_PATH=/app/data/processed/scaler.joblib \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

RUN groupadd --gid 1000 ${APP_USER} && \
    useradd --uid 1000 --gid ${APP_USER} --shell /bin/bash --create-home ${APP_USER}

RUN mkdir -p ${APP_HOME} \
    ${APP_HOME}/data/raw \
    ${APP_HOME}/data/processed \
    ${APP_HOME}/models \
    ${APP_HOME}/mlruns \
    ${APP_HOME}/logs \
    ${APP_HOME}/config \
    ${PROMETHEUS_MULTIPROC_DIR} && \
    chown -R ${APP_USER}:${APP_USER} ${APP_HOME} ${PROMETHEUS_MULTIPROC_DIR}

WORKDIR ${APP_HOME}

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=${APP_USER}:${APP_USER} src/ ${APP_HOME}/src/
COPY --chown=${APP_USER}:${APP_USER} api/ ${APP_HOME}/api/
COPY --chown=${APP_USER}:${APP_USER} config/ ${APP_HOME}/config/

USER ${APP_USER}

EXPOSE ${APP_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
