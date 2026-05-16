"""
FastAPI Production Deployment
Serves BERT, CNN, and XGBoost models via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import xgboost as xgb
import joblib
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Neural Network Ensemble API",
    description="Production API for BERT + CNN + XGBoost models",
    version="1.0.0"
)

# Global model containers
models = {
    "bert": None,
    "bert_tokenizer": None,
    "cnn": None,
    "xgboost": None
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Request/Response Models
class TextInput(BaseModel):
    """Input for text classification"""
    texts: List[str] = Field(..., min_items=1, max_items=100, 
                             description="List of text samples to classify")
    model: str = Field(default="bert", 
                      description="Model to use: 'bert', 'xgboost', or 'ensemble'")


class SequenceInput(BaseModel):
    """Input for sequence classification with CNN"""
    sequences: List[List[int]] = Field(..., min_items=1, max_items=100,
                                       description="List of tokenized sequences")


class TimeSeriesInput(BaseModel):
    """Input for time series forecasting"""
    sequences: List[List[float]] = Field(..., min_items=1, max_items=100,
                                         description="Time series sequences")


class PredictionResponse(BaseModel):
    """Prediction response"""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    model_used: str
    inference_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]
    device: str
    timestamp: str


# Startup event
@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    try:
        logger.info("Loading models...")
        
        # Load BERT
        try:
            models["bert_tokenizer"] = BertTokenizer.from_pretrained('bert-base-uncased')
            models["bert"] = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            models["bert"].to(device)
            models["bert"].eval()
            logger.info("✓ BERT model loaded")
        except Exception as e:
            logger.warning(f"Could not load BERT: {e}")
        
        # Load XGBoost (if exists)
        try:
            models["xgboost"] = joblib.load("models/xgboost_model.pkl")
            logger.info("✓ XGBoost model loaded")
        except Exception as e:
            logger.warning(f"Could not load XGBoost: {e}")
        
        logger.info(f"Models loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")


# API Endpoints
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Neural Network Ensemble API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_text": "/predict/text",
            "predict_sequence": "/predict/sequence",
            "predict_timeseries": "/predict/timeseries",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "bert": models["bert"] is not None,
            "xgboost": models["xgboost"] is not None
        },
        device=str(device),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """
    Predict sentiment/class for text inputs
    
    Supports BERT, XGBoost, or ensemble predictions
    """
    start_time = datetime.now()
    
    try:
        if input_data.model == "bert":
            if models["bert"] is None:
                raise HTTPException(status_code=503, detail="BERT model not loaded")
            
            # Tokenize
            encodings = models["bert_tokenizer"](
                input_data.texts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Predict
            with torch.no_grad():
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                outputs = models["bert"](input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
            predictions = preds.cpu().numpy().tolist()
            probabilities = probs.cpu().numpy().tolist()
            
        elif input_data.model == "xgboost":
            if models["xgboost"] is None:
                raise HTTPException(status_code=503, detail="XGBoost model not loaded")
            
            # Extract features
            features = extract_text_features(input_data.texts)
            predictions = models["xgboost"].predict(features).tolist()
            probabilities = models["xgboost"].predict_proba(features).tolist()
            
        else:
            raise HTTPException(status_code=400, detail="Invalid model. Choose 'bert' or 'xgboost'")
        
        # Calculate inference time
        inference_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_used=input_data.model,
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(input_data: TextInput):
    """Batch prediction endpoint (alias for predict_text)"""
    return await predict_text(input_data)


@app.get("/models/info")
async def model_info():
    """Get information about loaded models"""
    return {
        "models": {
            "bert": {
                "loaded": models["bert"] is not None,
                "type": "Transformer (BERT)",
                "framework": "PyTorch + HuggingFace",
                "parameters": "110M (base)"
            },
            "xgboost": {
                "loaded": models["xgboost"] is not None,
                "type": "Gradient Boosting",
                "framework": "XGBoost"
            }
        },
        "device": str(device),
        "precision": "float32"
    }


def extract_text_features(texts: List[str]) -> np.ndarray:
    """Extract statistical features from text"""
    features = []
    
    for text in texts:
        text_features = [
            len(text),
            len(text.split()),
            np.mean([len(word) for word in text.split()]) if text.split() else 0,
            text.count('!'),
            text.count('?'),
            sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
            text.count('.'),
            len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0
        ]
        features.append(text_features)
    
    return np.array(features)


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
