"""
Neural Network Training Pipeline
Transformer (BERT) + CNN + Time Series with MLflow Tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class CNNClassifier(nn.Module):
    """1D CNN for sequence classification"""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, num_classes: int = 2):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class TimeSeriesLSTM(nn.Module):
    """LSTM for time series forecasting"""
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class MLPipeline:
    """Unified ML Pipeline with BERT + CNN + Time Series"""
    
    def __init__(self, experiment_name: str = "neural_network_pipeline"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        mlflow.set_experiment(experiment_name)
    
    def train_bert_with_lora(
        self, 
        train_texts: List[str], 
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        num_epochs: int = 3,
        batch_size: int = 16
    ) -> Dict:
        """Fine-tune BERT with PEFT/LoRA"""
        
        with mlflow.start_run(run_name="BERT_LoRA"):
            # Log parameters
            mlflow.log_params({
                "model": "bert-base-uncased",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1
            })
            
            # Initialize tokenizer and model
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', 
                num_labels=2
            )
            
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "value"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            model.to(self.device)
            
            # Prepare data
            train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
            val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
            
            train_dataset = TensorDataset(
                train_encodings['input_ids'],
                train_encodings['attention_mask'],
                torch.tensor(train_labels)
            )
            val_dataset = TensorDataset(
                val_encodings['input_ids'],
                val_encodings['attention_mask'],
                torch.tensor(val_labels)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Optimizer and scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=total_steps
            )
            
            # Training loop
            best_val_acc = 0
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                for batch in train_loader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_preds, val_true = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                        outputs = model(input_ids, attention_mask=attention_mask)
                        preds = torch.argmax(outputs.logits, dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_true.extend(labels.cpu().numpy())
                
                val_acc = accuracy_score(val_true, val_preds)
                avg_train_loss = train_loss / len(train_loader)
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch
                }, step=epoch)
                
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Log model
            mlflow.pytorch.log_model(model, "bert_lora_model")
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            
            print(f"\nBERT LoRA - Best Validation Accuracy: {best_val_acc:.4f}")
            return {
                "model": "BERT_LoRA",
                "best_accuracy": best_val_acc,
                "final_train_loss": avg_train_loss
            }
    
    def train_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        vocab_size: int = 10000,
        num_epochs: int = 10,
        batch_size: int = 32
    ) -> Dict:
        """Train 1D CNN classifier"""
        
        with mlflow.start_run(run_name="CNN_Classifier"):
            mlflow.log_params({
                "model": "CNN_1D",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "vocab_size": vocab_size
            })
            
            # Prepare data
            train_dataset = TensorDataset(
                torch.LongTensor(X_train),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.LongTensor(X_val),
                torch.LongTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            model = CNNClassifier(vocab_size=vocab_size).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            best_val_acc = 0
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_preds, val_true = [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        outputs = model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_true.extend(labels.numpy())
                
                val_acc = accuracy_score(val_true, val_preds)
                avg_train_loss = train_loss / len(train_loader)
                
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch
                }, step=epoch)
                
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            mlflow.pytorch.log_model(model, "cnn_model")
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            
            print(f"\nCNN - Best Validation Accuracy: {best_val_acc:.4f}")
            return {
                "model": "CNN_1D",
                "best_accuracy": best_val_acc,
                "final_train_loss": avg_train_loss
            }
    
    def train_time_series(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 50,
        batch_size: int = 64
    ) -> Dict:
        """Train LSTM for time series forecasting"""
        
        with mlflow.start_run(run_name="TimeSeries_LSTM"):
            mlflow.log_params({
                "model": "LSTM",
                "epochs": num_epochs,
                "batch_size": batch_size,
                "hidden_size": 64,
                "num_layers": 2
            })
            
            # Prepare data
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            model = TimeSeriesLSTM().to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            best_val_mse = float('inf')
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                mlflow.log_metrics({
                    "train_mse": avg_train_loss,
                    "val_mse": avg_val_loss,
                    "epoch": epoch
                }, step=epoch)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train MSE: {avg_train_loss:.6f}, Val MSE: {avg_val_loss:.6f}")
                
                if avg_val_loss < best_val_mse:
                    best_val_mse = avg_val_loss
            
            mlflow.pytorch.log_model(model, "lstm_model")
            mlflow.log_metric("best_val_mse", best_val_mse)
            
            print(f"\nLSTM - Best Validation MSE: {best_val_mse:.6f}")
            return {
                "model": "LSTM",
                "best_mse": best_val_mse,
                "final_train_loss": avg_train_loss
            }


def generate_sample_data():
    """Generate sample datasets for demonstration"""
    
    # Text classification data for BERT
    texts = [
        "This product is amazing, highly recommend!",
        "Terrible experience, waste of money.",
        "Great quality and fast shipping.",
        "Not what I expected, very disappointing.",
        "Excellent service and support team.",
        "Poor quality, broke after one use.",
    ] * 50  # Replicate for more data
    labels = [1, 0, 1, 0, 1, 0] * 50
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Sequence data for CNN (simulated tokenized text)
    X_seq = np.random.randint(0, 10000, size=(1000, 50))
    y_seq = np.random.randint(0, 2, size=1000)
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    # Time series data
    time_steps = 20
    X_ts = np.array([np.sin(np.linspace(0, 3*np.pi, time_steps) + i*0.1).reshape(-1, 1) 
                     for i in range(1000)])
    y_ts = np.array([np.sin(3*np.pi + i*0.1) for i in range(1000)]).reshape(-1, 1)
    X_train_ts, X_val_ts, y_train_ts, y_val_ts = train_test_split(
        X_ts, y_ts, test_size=0.2, random_state=42
    )
    
    return {
        "bert": (train_texts, train_labels, val_texts, val_labels),
        "cnn": (X_train_seq, y_train_seq, X_val_seq, y_val_seq),
        "timeseries": (X_train_ts, y_train_ts, X_val_ts, y_val_ts)
    }


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MLPipeline(experiment_name="neural_network_pipeline")
    
    # Generate sample data
    print("Generating sample datasets...")
    data = generate_sample_data()
    
    # Train all models
    results = []
    
    print("\n" + "="*60)
    print("Training BERT with LoRA Fine-tuning")
    print("="*60)
    bert_results = pipeline.train_bert_with_lora(*data["bert"], num_epochs=2)
    results.append(bert_results)
    
    print("\n" + "="*60)
    print("Training CNN Classifier")
    print("="*60)
    cnn_results = pipeline.train_cnn(*data["cnn"], num_epochs=5)
    results.append(cnn_results)
    
    print("\n" + "="*60)
    print("Training Time Series LSTM")
    print("="*60)
    ts_results = pipeline.train_time_series(*data["timeseries"], num_epochs=30)
    results.append(ts_results)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING PIPELINE SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['model']}:")
        for key, value in result.items():
            if key != 'model':
                print(f"  {key}: {value:.4f}")
    
    print("\n✅ All models trained successfully!")
    print("📊 Check MLflow UI: mlflow ui --port 5000")
