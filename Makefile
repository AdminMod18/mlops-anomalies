.PHONY: help install train predict api docker-build docker-run k8s-deploy k8s-delete test clean

IMAGE_NAME = adminmod18/mlops-anomalies
IMAGE_TAG = latest
DOCKER_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install dev dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort

data-generate: ## Generate synthetic data
	python -c "from src.utils import generate_synthetic_data; generate_synthetic_data('data/raw/anomaly_data.csv', n_samples=10000)"

data-preprocess: ## Preprocess data
	python src/preprocess.py

train: ## Train the model
	python src/train.py

predict: ## Run prediction example
	python src/predict.py

mlflow-ui: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

api: ## Run FastAPI locally
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod: ## Run FastAPI in production mode
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) .

docker-run: ## Run Docker container
	docker run -d -p 8000:8000 --name mlops-anomalies $(DOCKER_IMAGE)

docker-stop: ## Stop Docker container
	docker stop mlops-anomalies || true
	docker rm mlops-anomalies || true

docker-push: ## Push Docker image to registry
	docker push $(DOCKER_IMAGE)

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f kubernetes/

k8s-delete: ## Delete from Kubernetes
	kubectl delete -f kubernetes/ || true

k8s-status: ## Show Kubernetes status
	kubectl get pods,svc,hpa -l app=mlops-anomalies

k8s-logs: ## Show Kubernetes logs
	kubectl logs -l app=mlops-anomalies -f

k8s-port-forward: ## Port forward to local
	kubectl port-forward svc/mlops-anomalies-service 8000:8000

prometheus-port-forward: ## Port forward Prometheus
	kubectl port-forward svc/prometheus-service 9090:9090

grafana-port-forward: ## Port forward Grafana
	kubectl port-forward svc/grafana-service 3000:3000

test: ## Run tests
	pytest tests/ -v --cov=src --cov=api --cov-report=html

clean: ## Clean up generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

pipeline: data-generate data-preprocess train ## Run full ML pipeline
	@echo "Full pipeline completed!"

setup: install data-generate data-preprocess train ## Complete setup
	@echo "Setup completed! Run 'make api' to start the API"
