"""
Unit Tests for Neural Network Pipeline
Tests BERT, CNN, LSTM, and API components
"""

import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_pipeline import CNNClassifier, TimeSeriesLSTM, MLPipeline
from api_server import app


class TestCNNClassifier:
    """Test CNN classifier"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = CNNClassifier(vocab_size=1000, embedding_dim=128, num_classes=2)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with random input"""
        model = CNNClassifier(vocab_size=1000, embedding_dim=128, num_classes=2)
        batch_size, seq_len = 8, 50
        
        # Create random input
        x = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 2)
        assert not torch.isnan(output).any()
    
    def test_output_range(self):
        """Test output logits are reasonable"""
        model = CNNClassifier(vocab_size=1000, embedding_dim=128, num_classes=2)
        x = torch.randint(0, 1000, (4, 50))
        
        output = model(x)
        
        # Logits should be finite
        assert torch.isfinite(output).all()


class TestTimeSeriesLSTM:
    """Test LSTM for time series"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = TimeSeriesLSTM(input_size=1, hidden_size=64, num_layers=2)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = TimeSeriesLSTM(input_size=1, hidden_size=64, num_layers=2)
        batch_size, seq_len, input_size = 16, 20, 1
        
        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
    
    def test_multiple_features(self):
        """Test with multiple input features"""
        model = TimeSeriesLSTM(input_size=5, hidden_size=64, num_layers=2)
        x = torch.randn(8, 20, 5)
        
        output = model(x)
        assert output.shape == (8, 1)


class TestMLPipeline:
    """Test ML Pipeline"""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized"""
        pipeline = MLPipeline(experiment_name="test_experiment")
        assert pipeline is not None
        assert pipeline.device is not None
    
    def test_device_selection(self):
        """Test device is selected correctly"""
        pipeline = MLPipeline()
        device = pipeline.device
        
        assert device.type in ['cuda', 'cpu']


class TestAPI:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, client):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["status"] == "healthy"
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "device" in data
    
    def test_predict_text_validation(self, client):
        """Test input validation"""
        # Empty texts should fail
        response = client.post(
            "/predict/text",
            json={"texts": [], "model": "bert"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_text_invalid_model(self, client):
        """Test invalid model selection"""
        response = client.post(
            "/predict/text",
            json={"texts": ["test"], "model": "invalid_model"}
        )
        # Should return error for invalid model
        assert response.status_code in [400, 503]


class TestDataGeneration:
    """Test data generation utilities"""
    
    def test_sample_data_shapes(self):
        """Test generated data has correct shapes"""
        from train_pipeline import generate_sample_data
        
        data = generate_sample_data()
        
        # Check BERT data
        train_texts, train_labels, val_texts, val_labels = data["bert"]
        assert len(train_texts) == len(train_labels)
        assert len(val_texts) == len(val_labels)
        assert all(isinstance(text, str) for text in train_texts)
        
        # Check CNN data
        X_train, y_train, X_val, y_val = data["cnn"]
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]
        assert X_train.ndim == 2
        
        # Check time series data
        X_train_ts, y_train_ts, X_val_ts, y_val_ts = data["timeseries"]
        assert X_train_ts.shape[0] == y_train_ts.shape[0]
        assert X_val_ts.shape[0] == y_val_ts.shape[0]
        assert X_train_ts.ndim == 3


class TestEnsemble:
    """Test ensemble components"""
    
    def test_feature_extraction(self):
        """Test text feature extraction"""
        from ensemble_pipeline import FeatureExtractor
        
        texts = [
            "This is a test!",
            "Another sample text.",
            "SHORT"
        ]
        
        features = FeatureExtractor.extract_text_features(texts)
        
        assert features.shape[0] == len(texts)
        assert features.shape[1] == 8  # 8 features per text
        assert not np.isnan(features).any()
    
    def test_feature_ranges(self):
        """Test extracted features have reasonable ranges"""
        from ensemble_pipeline import FeatureExtractor
        
        texts = ["Test text with some words!"] * 10
        features = FeatureExtractor.extract_text_features(texts)
        
        # All features should be non-negative
        assert (features >= 0).all()
        
        # Uppercase ratio should be between 0 and 1
        uppercase_ratio = features[:, 5]
        assert (uppercase_ratio >= 0).all() and (uppercase_ratio <= 1).all()


def run_tests():
    """Run all tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
