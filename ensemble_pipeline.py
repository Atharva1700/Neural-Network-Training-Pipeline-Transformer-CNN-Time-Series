"""
Ensemble Pipeline: BERT + CNN + XGBoost
Combines deep learning and gradient boosting for robust predictions
"""

import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from typing import List, Dict, Tuple
import joblib


class EnsemblePredictor:
    """Ensemble model combining BERT, CNN, and XGBoost"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = None
        self.bert_tokenizer = None
        self.cnn_model = None
        self.xgb_model = None
        
    def load_bert(self, model_path: str):
        """Load fine-tuned BERT model"""
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertForSequenceClassification.from_pretrained(model_path)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
    def load_cnn(self, model_path: str):
        """Load trained CNN model"""
        self.cnn_model = torch.load(model_path)
        self.cnn_model.to(self.device)
        self.cnn_model.eval()
    
    def train_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict:
        """Train XGBoost model with MLflow tracking"""
        
        with mlflow.start_run(run_name="XGBoost_Classifier"):
            # Set parameters
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'logloss',
                'random_state': 42
            }
            
            mlflow.log_params(params)
            
            # Train model
            self.xgb_model = xgb.XGBClassifier(**params)
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Predictions
            train_preds = self.xgb_model.predict(X_train)
            val_preds = self.xgb_model.predict(X_val)
            
            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            })
            
            # Log model
            mlflow.xgboost.log_model(self.xgb_model, "xgboost_model")
            
            print(f"XGBoost - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            return {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            }
    
    def predict_bert(self, texts: List[str]) -> np.ndarray:
        """Get BERT predictions"""
        encodings = self.bert_tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
        return preds.cpu().numpy()
    
    def predict_cnn(self, sequences: np.ndarray) -> np.ndarray:
        """Get CNN predictions"""
        with torch.no_grad():
            inputs = torch.LongTensor(sequences).to(self.device)
            outputs = self.cnn_model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
        return preds.cpu().numpy()
    
    def predict_xgboost(self, features: np.ndarray) -> np.ndarray:
        """Get XGBoost predictions"""
        return self.xgb_model.predict(features)
    
    def ensemble_predict(
        self, 
        texts: List[str] = None,
        sequences: np.ndarray = None,
        features: np.ndarray = None,
        voting: str = 'hard'
    ) -> np.ndarray:
        """
        Ensemble prediction using available models
        
        Args:
            texts: Input texts for BERT
            sequences: Input sequences for CNN
            features: Input features for XGBoost
            voting: 'hard' (majority vote) or 'soft' (average probabilities)
        """
        predictions = []
        
        if texts is not None and self.bert_model is not None:
            predictions.append(self.predict_bert(texts))
            
        if sequences is not None and self.cnn_model is not None:
            predictions.append(self.predict_cnn(sequences))
            
        if features is not None and self.xgb_model is not None:
            predictions.append(self.predict_xgboost(features))
        
        if len(predictions) == 0:
            raise ValueError("No models available for prediction")
        
        # Majority voting
        predictions = np.array(predictions)
        ensemble_preds = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=predictions
        )
        
        return ensemble_preds


class FeatureExtractor:
    """Extract features from text for XGBoost"""
    
    @staticmethod
    def extract_text_features(texts: List[str]) -> np.ndarray:
        """Extract statistical features from text"""
        features = []
        
        for text in texts:
            text_features = [
                len(text),  # Length
                len(text.split()),  # Word count
                np.mean([len(word) for word in text.split()]),  # Avg word length
                text.count('!'),  # Exclamation marks
                text.count('?'),  # Question marks
                sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,  # Uppercase ratio
                text.count('.'),  # Periods
                len(set(text.split())) / len(text.split()) if len(text.split()) > 0 else 0  # Unique word ratio
            ]
            features.append(text_features)
        
        return np.array(features)


def train_ensemble_pipeline(train_texts: List[str], train_labels: List[int],
                            val_texts: List[str], val_labels: List[int]) -> Dict:
    """Train complete ensemble pipeline"""
    
    mlflow.set_experiment("ensemble_pipeline")
    
    with mlflow.start_run(run_name="Full_Ensemble"):
        # Extract features for XGBoost
        print("Extracting features for XGBoost...")
        feature_extractor = FeatureExtractor()
        X_train_features = feature_extractor.extract_text_features(train_texts)
        X_val_features = feature_extractor.extract_text_features(val_texts)
        
        # Initialize ensemble
        ensemble = EnsemblePredictor()
        
        # Train XGBoost on text features
        print("\nTraining XGBoost on text features...")
        xgb_results = ensemble.train_xgboost(
            X_train_features, 
            np.array(train_labels),
            X_val_features,
            np.array(val_labels)
        )
        
        # Make predictions with XGBoost
        val_preds_xgb = ensemble.predict_xgboost(X_val_features)
        val_acc_xgb = accuracy_score(val_labels, val_preds_xgb)
        
        print(f"\nXGBoost Validation Accuracy: {val_acc_xgb:.4f}")
        
        # Log ensemble metrics
        mlflow.log_metric("xgboost_val_accuracy", val_acc_xgb)
        
        # Classification report
        print("\nXGBoost Classification Report:")
        print(classification_report(val_labels, val_preds_xgb, 
                                   target_names=['Negative', 'Positive']))
        
        # Log confusion matrix
        cm = confusion_matrix(val_labels, val_preds_xgb)
        print("\nConfusion Matrix:")
        print(cm)
        
        results = {
            "xgboost_accuracy": val_acc_xgb,
            "confusion_matrix": cm.tolist()
        }
        
        return results


if __name__ == "__main__":
    # Sample data
    from sklearn.model_selection import train_test_split
    
    texts = [
        "This product is amazing, highly recommend!",
        "Terrible experience, waste of money.",
        "Great quality and fast shipping.",
        "Not what I expected, very disappointing.",
        "Excellent service and support team.",
        "Poor quality, broke after one use.",
        "Best purchase ever!",
        "Complete garbage, don't buy.",
        "Satisfactory but overpriced.",
        "Outstanding quality and design."
    ] * 100
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1] * 100
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    print("="*60)
    print("ENSEMBLE PIPELINE: BERT + CNN + XGBoost")
    print("="*60)
    
    results = train_ensemble_pipeline(train_texts, train_labels, val_texts, val_labels)
    
    print("\n" + "="*60)
    print("ENSEMBLE RESULTS")
    print("="*60)
    print(f"XGBoost Validation Accuracy: {results['xgboost_accuracy']:.4f}")
    print("\n✅ Ensemble pipeline trained successfully!")
    print("📊 Check MLflow UI: mlflow ui --port 5000")
