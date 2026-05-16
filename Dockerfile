# Multi-stage Docker build for Neural Network Ensemble API
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY train_pipeline.py .
COPY ensemble_pipeline.py .
COPY api_server.py .

# Create directories for models and MLflow
RUN mkdir -p models mlruns

# Expose ports
EXPOSE 8000 5000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (API server)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# Training pipeline: python train_pipeline.py
# Ensemble pipeline: python ensemble_pipeline.py
# MLflow UI: mlflow ui --host 0.0.0.0 --port 5000
