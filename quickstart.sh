#!/bin/bash

# Neural Network Pipeline - Quick Start Script
# Automates setup, training, and deployment

set -e  # Exit on error

echo "=================================================="
echo "Neural Network Pipeline - Quick Start"
echo "=================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Check Python installation
print_step "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
print_success "Found $PYTHON_VERSION"

# Create virtual environment
print_step "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Install dependencies
print_step "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
print_success "Dependencies installed"

# Create necessary directories
print_step "Creating project directories..."
mkdir -p models mlruns cache
print_success "Directories created"

# Menu for user choice
echo ""
echo "=================================================="
echo "What would you like to do?"
echo "=================================================="
echo "1) Train all models (BERT, CNN, LSTM)"
echo "2) Train ensemble (BERT + XGBoost)"
echo "3) Start API server"
echo "4) Run API client examples"
echo "5) Start MLflow UI"
echo "6) Run full pipeline (train + API)"
echo "7) Docker deployment"
echo "8) Exit"
echo ""
read -p "Enter your choice [1-8]: " choice

case $choice in
    1)
        print_step "Training all models..."
        python train_pipeline.py
        print_success "Training completed!"
        print_info "Check MLflow UI: mlflow ui --port 5000"
        ;;
    2)
        print_step "Training ensemble pipeline..."
        python ensemble_pipeline.py
        print_success "Ensemble training completed!"
        ;;
    3)
        print_step "Starting API server..."
        print_info "API will be available at http://localhost:8000"
        print_info "API docs at http://localhost:8000/docs"
        print_info "Press Ctrl+C to stop"
        python api_server.py
        ;;
    4)
        print_step "Running API client examples..."
        print_info "Make sure API server is running (python api_server.py)"
        sleep 2
        python client_example.py
        ;;
    5)
        print_step "Starting MLflow UI..."
        print_info "MLflow UI will be available at http://localhost:5000"
        print_info "Press Ctrl+C to stop"
        mlflow ui --port 5000
        ;;
    6)
        print_step "Running full pipeline..."
        
        # Train models
        print_step "Step 1/4: Training BERT, CNN, LSTM..."
        python train_pipeline.py
        print_success "Models trained"
        
        # Train ensemble
        print_step "Step 2/4: Training ensemble..."
        python ensemble_pipeline.py
        print_success "Ensemble trained"
        
        # Start MLflow in background
        print_step "Step 3/4: Starting MLflow UI..."
        mlflow ui --port 5000 &
        MLFLOW_PID=$!
        sleep 3
        print_success "MLflow UI running at http://localhost:5000"
        
        # Start API server
        print_step "Step 4/4: Starting API server..."
        print_info "API available at http://localhost:8000"
        print_info "Press Ctrl+C to stop all services"
        
        # Cleanup on exit
        trap "kill $MLFLOW_PID 2>/dev/null" EXIT
        
        python api_server.py
        ;;
    7)
        print_step "Docker deployment..."
        
        # Check Docker installation
        if ! command -v docker &> /dev/null; then
            echo "Docker is not installed. Please install Docker first."
            exit 1
        fi
        
        print_info "Building Docker image..."
        docker-compose build
        
        print_info "Starting services..."
        docker-compose up -d
        
        print_success "Services started!"
        print_info "API: http://localhost:8000"
        print_info "MLflow: http://localhost:5000"
        print_info ""
        print_info "View logs: docker-compose logs -f"
        print_info "Stop services: docker-compose down"
        ;;
    8)
        print_info "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
print_success "Done!"
