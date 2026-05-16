.PHONY: help install train api mlflow test docker clean

help:
	@echo "Neural Network Pipeline - Makefile Commands"
	@echo "============================================"
	@echo "make install      - Install dependencies"
	@echo "make train        - Train all models"
	@echo "make ensemble     - Train ensemble pipeline"
	@echo "make api          - Start API server"
	@echo "make mlflow       - Start MLflow UI"
	@echo "make test         - Run tests"
	@echo "make docker-build - Build Docker images"
	@echo "make docker-up    - Start Docker services"
	@echo "make docker-down  - Stop Docker services"
	@echo "make clean        - Clean generated files"
	@echo "make all          - Install, train, and start services"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

train:
	@echo "Training all models (BERT, CNN, LSTM)..."
	python train_pipeline.py

ensemble:
	@echo "Training ensemble pipeline..."
	python ensemble_pipeline.py

api:
	@echo "Starting API server..."
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	python api_server.py

mlflow:
	@echo "Starting MLflow UI..."
	@echo "MLflow: http://localhost:5000"
	mlflow ui --port 5000

test:
	@echo "Running tests..."
	python test_pipeline.py

client:
	@echo "Running API client examples..."
	python client_example.py

docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d
	@echo "API: http://localhost:8000"
	@echo "MLflow: http://localhost:5000"

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	@echo "Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf mlruns
	@echo "Clean complete!"

all: install train ensemble
	@echo "Running full pipeline..."
	@echo "Starting services..."
	@echo "MLflow in background..."
	mlflow ui --port 5000 &
	@echo "Starting API..."
	python api_server.py
