"""Unit tests for FastAPI endpoints."""
import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_endpoint(self):
        response = client.get("/ready")
        assert response.status_code in [200, 503]


class TestPredictionEndpoints:
    def test_predict_single(self):
        payload = {"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "anomaly_score" in data
            assert "is_anomaly" in data

    def test_predict_batch(self):
        payload = {
            "samples": [
                {"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
                {"features": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]}
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code in [200, 503]

    def test_predict_invalid_features(self):
        payload = {"features": [0.1, 0.2, 0.3]}
        response = client.post("/predict", json=payload)
        assert response.status_code in [400, 422, 503]


class TestMetricsEndpoint:
    def test_metrics_endpoint(self):
        response = client.get("/metrics")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
