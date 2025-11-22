# Checklist para Exposición del Proyecto MLOps

## Resumen Ejecutivo

**Proyecto**: Pipeline MLOps para Detección de Anomalías en Tiempo Real

**Stack**: Python 3.10, FastAPI, Isolation Forest, Docker, Kubernetes, MLflow, DVC, Prometheus, Grafana, GitHub Actions

## Arquitectura

```
GitHub → CI/CD Pipeline → Docker Hub → Kubernetes
                ↓
            MLflow (Tracking)
                ↓
         FastAPI (2 réplicas)
                ↓
    Prometheus → Grafana (Dashboards)
```

## Checklist Pre-Presentación

### Una semana antes
- [ ] Ejecutar `./scripts/check_environment.sh`
- [ ] Verificar que el pipeline CI/CD pasa
- [ ] Preparar slides

### Un día antes
- [ ] Deploy fresco en Minikube
- [ ] Verificar todos los servicios
- [ ] Practicar demo

### Día de la presentación
- [ ] `minikube start`
- [ ] `kubectl port-forward svc/mlops-anomalies-service 8000:8000`
- [ ] `kubectl port-forward svc/prometheus-service 9090:9090`
- [ ] `kubectl port-forward svc/grafana-service 3000:3000`
- [ ] Abrir tabs: localhost:8000/docs, localhost:9090, localhost:3000

## Comandos Demo

```bash
# Health check
curl http://localhost:8000/health

# Predicción normal
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'

# Predicción anómala
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [15.0, -12.0, 20.0, -18.0, 25.0, -22.0, 18.0, -15.0, 12.0, -10.0]}'

# Ver pods
kubectl get pods -l app=mlops-anomalies

# Ver HPA
kubectl get hpa
```

## Preguntas del Profesor

**¿Por qué Isolation Forest?**
> Es no supervisado, escala bien, y detecta anomalías naturalmente por aislamiento.

**¿Cómo garantizan reproducibilidad?**
> MLflow (tracking), DVC (datos), Git (código), Docker (entorno), random_state=42.

**¿Qué pasa si el modelo falla?**
> Readiness probe, múltiples réplicas, rollback automático, alertas en Prometheus.

**¿Cómo escala?**
> HPA: 2-10 réplicas, escala al 70% CPU / 80% memoria.

## Métricas a Mostrar

| Métrica | Query |
|---------|-------|
| Latencia p95 | `histogram_quantile(0.95, rate(anomaly_api_request_latency_seconds_bucket[5m]))` |
| Request Rate | `rate(anomaly_api_requests_total[5m])` |
| Anomaly Rate | `rate(anomaly_predictions_total{result="anomaly"}[5m]) / rate(anomaly_predictions_total[5m])` |

## Grafana
- Usuario: admin
- Password: admin123
