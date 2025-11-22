"""
FastAPI Application for MLOps Anomaly Detection.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.model_loader import get_model_loader

# API Settings
API_TITLE = "MLOps Anomaly Detection API"
API_DESCRIPTION = """
## Anomaly Detection API

Real-time anomaly detection using Isolation Forest.

### Features:
- Single and batch predictions
- Kubernetes-ready health endpoints
- Prometheus metrics
"""
API_VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Prometheus Metrics
REQUEST_COUNT = Counter('anomaly_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('anomaly_api_request_latency_seconds', 'Request latency', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('anomaly_predictions_total', 'Total predictions', ['result'])
PREDICTION_LATENCY = Histogram('anomaly_prediction_latency_seconds', 'Prediction latency')
ANOMALY_SCORE = Histogram('anomaly_score_distribution', 'Anomaly scores', buckets=[-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5])
MODEL_LOADED = Gauge('anomaly_model_loaded', 'Model loaded status')
ACTIVE_REQUESTS = Gauge('anomaly_active_requests', 'Active requests')


# Pydantic Models
class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_length=10, max_length=10, description="List of 10 features")

    @validator('features')
    def validate_features(cls, v):
        if len(v) != 10:
            raise ValueError('Must provide exactly 10 features')
        if any(np.isnan(x) or np.isinf(x) for x in v):
            raise ValueError('Features cannot contain NaN or Inf')
        return v


class PredictionOutput(BaseModel):
    prediction: int
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    label: str
    inference_time_ms: float
    timestamp: str


class BatchPredictionInput(BaseModel):
    samples: List[PredictionInput] = Field(..., min_length=1, max_length=1000)


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_samples: int
    anomalies_detected: int
    normal_detected: int
    anomaly_ratio: float
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    n_features: int
    loaded_at: Optional[str]
    model_path: Optional[str]


# Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting MLOps Anomaly Detection API...")
    loader = get_model_loader()
    if loader.initialize():
        MODEL_LOADED.set(1)
        logger.info("Model loaded successfully")
    else:
        MODEL_LOADED.set(0)
        logger.warning("Model not loaded")
    yield
    logger.info("Shutting down API...")
    MODEL_LOADED.set(0)


# FastAPI App
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
        return response
    finally:
        ACTIVE_REQUESTS.dec()


# Health Endpoints
@app.get("/", response_model=Dict[str, str], tags=["Root"])
async def root():
    return {"message": "MLOps Anomaly Detection API", "version": API_VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat(), version=API_VERSION)


@app.get("/ready", tags=["Health"])
async def readiness_check():
    loader = get_model_loader()
    if loader.is_ready:
        return JSONResponse(status_code=200, content={"status": "ready", "model_loaded": True})
    return JSONResponse(status_code=503, content={"status": "not_ready", "model_loaded": False})


# Prediction Endpoints
@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    loader = get_model_loader()
    if not loader.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        features = np.array(input_data.features, dtype=np.float64)
        start_time = time.time()
        prediction, score = loader.predict(features)
        inference_time = (time.time() - start_time) * 1000

        PREDICTION_LATENCY.observe(inference_time / 1000)
        ANOMALY_SCORE.observe(score)
        PREDICTION_COUNT.labels(result="anomaly" if prediction == 1 else "normal").inc()

        confidence = min(abs(score) * 50, 99.9)

        return PredictionOutput(
            prediction=prediction,
            is_anomaly=prediction == 1,
            anomaly_score=round(score, 6),
            confidence=round(confidence, 2),
            label="Anomaly" if prediction == 1 else "Normal",
            inference_time_ms=round(inference_time, 3),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    loader = get_model_loader()
    if not loader.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        batch_size = len(input_data.samples)
        features = np.array([s.features for s in input_data.samples], dtype=np.float64)

        start_time = time.time()
        predictions, scores = loader.predict_batch(features)
        total_inference_time = (time.time() - start_time) * 1000

        results = []
        for pred, score in zip(predictions, scores):
            confidence = min(abs(score) * 50, 99.9)
            ANOMALY_SCORE.observe(score)
            PREDICTION_COUNT.labels(result="anomaly" if pred == 1 else "normal").inc()

            results.append(PredictionOutput(
                prediction=int(pred),
                is_anomaly=bool(pred == 1),
                anomaly_score=round(float(score), 6),
                confidence=round(confidence, 2),
                label="Anomaly" if pred == 1 else "Normal",
                inference_time_ms=round(total_inference_time / batch_size, 3),
                timestamp=datetime.now().isoformat()
            ))

        anomalies_detected = int(sum(predictions))

        return BatchPredictionOutput(
            predictions=results,
            total_samples=batch_size,
            anomalies_detected=anomalies_detected,
            normal_detected=batch_size - anomalies_detected,
            anomaly_ratio=round(anomalies_detected / batch_size, 4),
            total_inference_time_ms=round(total_inference_time, 3)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Model Info
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    loader = get_model_loader()
    if not loader.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = loader.model_info or {}
    return ModelInfoResponse(
        model_name=info.get("model_name", "IsolationForest"),
        version=API_VERSION,
        n_features=info.get("n_features", 10),
        loaded_at=info.get("loaded_at"),
        model_path=info.get("model_path")
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    loader = get_model_loader()
    try:
        success = loader.load_model(force_reload=True)
        if success:
            MODEL_LOADED.set(1)
            return {"status": "success", "message": "Model reloaded"}
        MODEL_LOADED.set(0)
        raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


# Metrics Endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
