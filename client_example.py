"""
API Client Example
Test the deployed Neural Network Ensemble API
"""

import requests
import json
from typing import List, Dict
import time


class NeuralNetworkClient:
    """Client for interacting with Neural Network API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_text(self, texts: List[str], model: str = "bert") -> Dict:
        """
        Predict sentiment/class for text inputs
        
        Args:
            texts: List of text strings
            model: Model to use ('bert' or 'xgboost')
        """
        payload = {
            "texts": texts,
            "model": model
        }
        
        response = requests.post(
            f"{self.base_url}/predict/text",
            json=payload
        )
        
        return response.json()
    
    def model_info(self) -> Dict:
        """Get information about loaded models"""
        response = requests.get(f"{self.base_url}/models/info")
        return response.json()


def run_examples():
    """Run example predictions"""
    
    print("="*70)
    print("NEURAL NETWORK ENSEMBLE API - CLIENT EXAMPLES")
    print("="*70)
    
    # Initialize client
    client = NeuralNetworkClient()
    
    # Health check
    print("\n1. Health Check")
    print("-" * 70)
    try:
        health = client.health_check()
        print(json.dumps(health, indent=2))
    except Exception as e:
        print(f"❌ API not available: {e}")
        print("Make sure the API is running: python api_server.py")
        return
    
    # Model info
    print("\n2. Model Information")
    print("-" * 70)
    info = client.model_info()
    print(json.dumps(info, indent=2))
    
    # Test samples
    positive_samples = [
        "This product is absolutely amazing! Best purchase ever!",
        "Excellent quality and fast shipping. Highly recommend!",
        "Outstanding service and great value for money."
    ]
    
    negative_samples = [
        "Terrible experience. Complete waste of money.",
        "Poor quality, broke after one use. Very disappointed.",
        "Worst product ever. Do not buy this."
    ]
    
    all_samples = positive_samples + negative_samples
    
    # BERT predictions
    print("\n3. BERT Predictions")
    print("-" * 70)
    start = time.time()
    bert_result = client.predict_text(all_samples, model="bert")
    end = time.time()
    
    print(f"Inference Time: {bert_result['inference_time_ms']:.2f}ms")
    print(f"Total Time: {(end-start)*1000:.2f}ms")
    print("\nResults:")
    
    for i, (text, pred, prob) in enumerate(zip(
        all_samples, 
        bert_result['predictions'],
        bert_result['probabilities']
    )):
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = max(prob) * 100
        print(f"\n[{i+1}] {text[:60]}...")
        print(f"    → {sentiment} (confidence: {confidence:.1f}%)")
    
    # XGBoost predictions (if available)
    print("\n4. XGBoost Predictions (Feature-based)")
    print("-" * 70)
    try:
        xgb_result = client.predict_text(all_samples, model="xgboost")
        print(f"Inference Time: {xgb_result['inference_time_ms']:.2f}ms")
        print("\nResults:")
        
        for i, (text, pred) in enumerate(zip(all_samples, xgb_result['predictions'])):
            sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
            print(f"[{i+1}] {sentiment}")
    except Exception as e:
        print(f"XGBoost not available: {e}")
    
    # Batch prediction example
    print("\n5. Batch Prediction (100 samples)")
    print("-" * 70)
    
    batch_samples = [
        "Great product, very satisfied!",
        "Not worth the money.",
    ] * 50  # 100 samples
    
    start = time.time()
    batch_result = client.predict_text(batch_samples, model="bert")
    end = time.time()
    
    print(f"Total samples: {len(batch_samples)}")
    print(f"Inference time: {batch_result['inference_time_ms']:.2f}ms")
    print(f"Throughput: {len(batch_samples) / ((end-start)):.1f} samples/sec")
    
    # Accuracy on known labels
    true_labels = [1, 0] * 50
    predictions = batch_result['predictions']
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    print(f"Accuracy: {accuracy*100:.1f}%")
    
    print("\n" + "="*70)
    print("✅ All examples completed successfully!")
    print("="*70)


def benchmark_api():
    """Benchmark API performance"""
    
    print("\n" + "="*70)
    print("API PERFORMANCE BENCHMARK")
    print("="*70)
    
    client = NeuralNetworkClient()
    
    test_cases = [
        ("Single sample", ["This is a test"], 10),
        ("Small batch (5)", ["Test sample"] * 5, 10),
        ("Medium batch (20)", ["Test sample"] * 20, 5),
        ("Large batch (50)", ["Test sample"] * 50, 3),
    ]
    
    for name, samples, iterations in test_cases:
        times = []
        for _ in range(iterations):
            start = time.time()
            client.predict_text(samples, model="bert")
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        throughput = len(samples) / (avg_time / 1000)
        
        print(f"\n{name}:")
        print(f"  Avg latency: {avg_time:.2f}ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")


if __name__ == "__main__":
    print("\n🚀 Starting API Client Examples\n")
    
    # Run examples
    run_examples()
    
    # Run benchmark
    benchmark_api()
    
    print("\n💡 Tips:")
    print("  - Check MLflow UI: http://localhost:5000")
    print("  - API docs: http://localhost:8000/docs")
    print("  - Health check: http://localhost:8000/health")
