# MLOps Pipeline para Detección de Anomalías en Tiempo Real

[![CI/CD Pipeline](https://github.com/AdminMod18/mlops-anomalies/actions/workflows/mlops-pipeline.yml/badge.svg)](https://github.com/AdminMod18/mlops-anomalies/actions/workflows/mlops-pipeline.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.28-326CE5.svg)](https://kubernetes.io)

## Descripción

Pipeline MLOps completo para detección de anomalías en tiempo real usando **Isolation Forest**. Incluye entrenamiento automatizado, API REST, despliegue en Kubernetes, y monitorización con Prometheus/Grafana.

## Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| ML Model | Isolation Forest (scikit-learn) |
| API | FastAPI + Uvicorn |
| MLOps | MLflow + DVC |
| Containers | Docker |
| Orchestration | Kubernetes (Minikube) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

## Arquitectura

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   GitHub    │───▶│  CI/CD      │───▶│   Docker    │───▶│ Kubernetes  │
│  Repository │    │  Pipeline   │    │   Registry  │    │   Cluster   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                                      │
                         ▼                                      ▼
                  ┌─────────────┐                        ┌───────────┐
                  │   MLflow    │                        │ FastAPI   │
                  │  Tracking   │                        │   Pods    │
                  └─────────────┘                        └─────┬─────┘
                                                               │
                  ┌─────────────┐    ┌─────────────┐          │
                  │ Prometheus  │◀───│   Metrics   │◀─────────┘
                  └──────┬──────┘    └─────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   Grafana   │
                  └─────────────┘
```

## Instalación Rápida

### Opción 1: Local

```bash
# Clonar repositorio
git clone https://github.com/AdminMod18/mlops-anomalies.git
cd mlops-anomalies

# Setup completo
pip install -r requirements.txt
make setup

# Iniciar API
make api
```

### Opción 2: Docker

```bash
docker-compose up -d

# Acceder a:
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)
```

### Opción 3: Kubernetes (Minikube)

```bash
# Iniciar Minikube
minikube start --cpus=4 --memory=8192

# Build imagen
eval $(minikube docker-env)
docker build -t adminmod18/mlops-anomalies:latest .

# Deploy
kubectl apply -f kubernetes/

# Port forward
kubectl port-forward svc/mlops-anomalies-service 8000:8000
```

## Uso de la API

### Predicción Simple

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'
```

**Respuesta:**
```json
{
  "prediction": 0,
  "is_anomaly": false,
  "anomaly_score": 0.152843,
  "confidence": 85.5,
  "label": "Normal",
  "inference_time_ms": 2.541
}
```

### Predicción Batch

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
      {"features": [10.0, -8.0, 15.0, -12.0, 20.0, -15.0, 10.0, -10.0, 8.0, -7.0]}
    ]
  }'
```

## Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Info de la API |
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/predict` | POST | Predicción individual |
| `/predict/batch` | POST | Predicción en lote |
| `/model/info` | GET | Info del modelo |
| `/metrics` | GET | Métricas Prometheus |
| `/docs` | GET | Swagger UI |

## Estructura del Proyecto

```
mlops-anomalies/
├── api/                    # FastAPI application
├── src/                    # ML code (train, predict, preprocess)
├── kubernetes/             # K8s manifests
├── .github/workflows/      # CI/CD pipeline
├── tests/                  # Test suite
├── config/                 # Configuration
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── Dockerfile              # Multi-stage build
├── docker-compose.yml      # Local stack
├── Makefile                # Automation
├── dvc.yaml                # DVC pipeline
└── requirements.txt        # Dependencies
```

## Comandos Make

```bash
make install          # Instalar dependencias
make setup            # Setup completo (data + train)
make api              # Iniciar API local
make train            # Entrenar modelo
make test             # Ejecutar tests
make docker-build     # Build imagen Docker
make k8s-deploy       # Deploy en Kubernetes
make k8s-status       # Ver estado K8s
```

## Monitorización

### Grafana (http://localhost:3000)
- Usuario: `admin`
- Password: `admin123`

### Métricas Prometheus

```promql
# Latencia p95
histogram_quantile(0.95, rate(anomaly_api_request_latency_seconds_bucket[5m]))

# Request rate
rate(anomaly_api_requests_total[5m])

# Tasa de anomalías
rate(anomaly_predictions_total{result="anomaly"}[5m]) / rate(anomaly_predictions_total[5m])
```

## CI/CD Pipeline

El pipeline de GitHub Actions ejecuta:

1. **Lint** - Black, Flake8
2. **Test** - pytest con coverage
3. **Train** - Entrenamiento con MLflow
4. **Build** - Docker multi-platform
5. **Deploy** - Kubernetes
6. **Release** - GitHub releases

## Contribución

1. Fork el repositorio
2. Crear branch: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'feat: agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## Licencia

MIT License

---

**Autor**: MLOps Team
**Repositorio**: https://github.com/AdminMod18/mlops-anomalies
