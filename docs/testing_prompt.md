# Prompt para Pruebas Completas

Copia este prompt en Claude después de desplegar el proyecto:

---

```
Tengo desplegado el proyecto MLOps Anomaly Detection. Ejecuta las siguientes validaciones:

## 1. API Tests

# Health
curl http://localhost:8000/health

# Normal prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, -0.1, 0.3, 0.0, -0.2, 0.1, 0.2, -0.1, 0.0]}'

# Anomaly prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [15.0, -12.0, 20.0, -18.0, 25.0, -22.0, 18.0, -15.0, 12.0, -10.0]}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}, {"features": [10,-8,15,-12,20,-15,10,-10,8,-7]}]}'

## 2. Kubernetes

kubectl get pods -l app=mlops-anomalies
kubectl get svc
kubectl get hpa

## 3. Prometheus

curl 'http://localhost:9090/api/v1/query?query=anomaly_model_loaded'
curl 'http://localhost:9090/api/v1/query?query=anomaly_predictions_total'

## 4. Resiliencia

kubectl delete pod -l app=mlops-anomalies --wait=false
kubectl get pods -w

Genera un reporte con:
- Estado de cada componente (OK/ERROR)
- Latencia p95
- Tasa de anomalías
- Sugerencias de mejora
```
