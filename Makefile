# MLOps House Price Prediction Makefile

.PHONY: help install setup data train test api docker clean

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  setup       - Setup project environment"
	@echo "  data        - Generate synthetic data"
	@echo "  train       - Train the model"
	@echo "  test        - Run tests"
	@echo "  api         - Start API server"
	@echo "  docker      - Build Docker images"
	@echo "  clean       - Clean generated files"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Setup project environment
setup: install
	mkdir -p data/{raw,processed,models} logs artifacts
	cp .env.example .env
	@echo "Project setup complete. Edit .env file with your configuration."

# Generate synthetic data
data:
	python scripts/generate_data.py --samples 1000

# Train model
train:
	python scripts/train_model.py --optimize

# Run tests
test:
	pytest tests/ -v --cov=src

# Start API server
api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Build Docker images
docker:
	docker build -f docker/Dockerfile.api -t house-price-api:latest .
	docker build -f docker/Dockerfile.training -t house-price-training:latest .

# Start full stack with Docker Compose
docker-up:
	docker-compose -f docker/docker_compose.yml up -d

# Stop Docker stack
docker-down:
	docker-compose -f docker/docker_compose.yml down

# Clean generated files
clean:
	rm -rf data/processed/* data/models/* logs/* artifacts/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Format code
format:
	black src/ tests/ scripts/
	flake8 src/ tests/ scripts/

# Run MLflow UI
mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# Deploy to Kubernetes
k8s-deploy:
	kubectl apply -f kubernetes/

# Remove Kubernetes deployment
k8s-clean:
	kubectl delete -f kubernetes/