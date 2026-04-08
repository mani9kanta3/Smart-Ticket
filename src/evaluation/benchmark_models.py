"""
SmartTicket - Proper Benchmark Script
Compares all models on the SAME hardware (CPU) for a fair comparison.
This fixes the misleading comparison in the README.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import onnxruntime as ort
from transformers import DistilBertTokenizer, DistilBertModel


def benchmark_baseline(num_runs=200):
    """Benchmark TF-IDF + Logistic Regression on CPU."""
    print("\n[1/4] Benchmarking TF-IDF + Logistic Regression (CPU)...")
    
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    cat_model = joblib.load("models/baseline_category_model.joblib")
    
    sample_text = "my card payment was declined and I need help urgently"
    
    # Warmup
    for _ in range(20):
        X = tfidf.transform([sample_text])
        _ = cat_model.predict(X)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        X = tfidf.transform([sample_text])
        _ = cat_model.predict(X)
    elapsed_ms = (time.time() - start) / num_runs * 1000
    
    print(f"  Average: {elapsed_ms:.2f} ms/query")
    return elapsed_ms


def benchmark_bilstm_cpu(num_runs=200):
    """Benchmark BiLSTM on CPU (fair comparison with other CPU benchmarks)."""
    print("\n[2/4] Benchmarking BiLSTM (CPU)...")
    
    # Re-define model (must match training architecture)
    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256,
                     num_layers=2, num_categories=10, num_priorities=4,
                     dropout=0.3, fc_dropout=0.5, pad_idx=0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(
                input_size=embedding_dim, hidden_size=hidden_dim,
                num_layers=num_layers, batch_first=True,
                bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(fc_dropout)
            self.category_head = nn.Linear(hidden_dim * 2, num_categories)
            self.priority_head = nn.Linear(hidden_dim * 2, num_priorities)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, (hidden, cell) = self.lstm(embedded)
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            combined = torch.cat((forward_hidden, backward_hidden), dim=1)
            combined = self.dropout(combined)
            return self.category_head(combined), self.priority_head(combined)
    
    # Load model
    checkpoint = torch.load("models/bilstm_best.pth", map_location="cpu", weights_only=True)
    model = BiLSTMClassifier(vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Sample input
    sample_input = torch.randint(0, 100, (1, 64))
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(sample_input)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(sample_input)
    elapsed_ms = (time.time() - start) / num_runs * 1000
    
    print(f"  Average: {elapsed_ms:.2f} ms/query")
    return elapsed_ms


def benchmark_distilbert_pytorch_cpu(num_runs=200):
    """Benchmark DistilBERT (PyTorch) on CPU."""
    print("\n[3/4] Benchmarking DistilBERT PyTorch (CPU)...")
    
    class DistilBertMultiTask(nn.Module):
        def __init__(self, num_categories=10, num_priorities=4, dropout=0.3):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained("models/bert_finetuned")
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.category_head = nn.Sequential(
                nn.Linear(hidden_size, 256), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(256, num_categories),
            )
            self.priority_head = nn.Sequential(
                nn.Linear(hidden_size, 128), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(128, num_priorities),
            )
        
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            cls_output = self.dropout(cls_output)
            return self.category_head(cls_output), self.priority_head(cls_output)
    
    model = DistilBertMultiTask()
    checkpoint = torch.load("models/bert_finetuned/full_model.pth",
                            map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained("models/bert_finetuned")
    encoding = tokenizer(
        "my card payment was declined and I need help urgently",
        max_length=64, padding="max_length", truncation=True,
    )
    input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long)
    attention_mask = torch.tensor([encoding["attention_mask"]], dtype=torch.long)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(input_ids, attention_mask)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            model(input_ids, attention_mask)
    elapsed_ms = (time.time() - start) / num_runs * 1000
    
    print(f"  Average: {elapsed_ms:.2f} ms/query")
    return elapsed_ms


def benchmark_distilbert_onnx_cpu(num_runs=200):
    """Benchmark DistilBERT (ONNX) on CPU."""
    print("\n[4/4] Benchmarking DistilBERT ONNX (CPU)...")
    
    session = ort.InferenceSession(
        "models/smartticket_bert.onnx",
        providers=["CPUExecutionProvider"],
    )
    
    tokenizer = DistilBertTokenizer.from_pretrained("models/bert_finetuned")
    encoding = tokenizer(
        "my card payment was declined and I need help urgently",
        max_length=64, padding="max_length", truncation=True,
    )
    input_ids = np.array([encoding["input_ids"]], dtype=np.int64)
    attention_mask = np.array([encoding["attention_mask"]], dtype=np.int64)
    
    # Warmup
    for _ in range(20):
        session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    elapsed_ms = (time.time() - start) / num_runs * 1000
    
    print(f"  Average: {elapsed_ms:.2f} ms/query")
    return elapsed_ms


if __name__ == "__main__":
    print("=" * 60)
    print("SmartTicket - Fair CPU Benchmark")
    print("=" * 60)
    print("\nAll models benchmarked on the SAME CPU for fair comparison.")
    print("This fixes the misleading comparison in the original report.\n")
    
    baseline_ms = benchmark_baseline()
    bilstm_ms = benchmark_bilstm_cpu()
    bert_pytorch_ms = benchmark_distilbert_pytorch_cpu()
    bert_onnx_ms = benchmark_distilbert_onnx_cpu()
    
    print("\n" + "=" * 60)
    print("FAIR CPU BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n  {'Model':<30s} {'CPU Time (ms)':>15s} {'Speedup vs Baseline':>20s}")
    print(f"  {'-'*65}")
    print(f"  {'TF-IDF + LogReg':<30s} {baseline_ms:>14.2f}  {'1.0x':>20s}")
    print(f"  {'BiLSTM (PyTorch)':<30s} {bilstm_ms:>14.2f}  {f'{baseline_ms/bilstm_ms:.2f}x':>20s}")
    print(f"  {'DistilBERT (PyTorch)':<30s} {bert_pytorch_ms:>14.2f}  {f'{baseline_ms/bert_pytorch_ms:.2f}x':>20s}")
    print(f"  {'DistilBERT (ONNX)':<30s} {bert_onnx_ms:>14.2f}  {f'{baseline_ms/bert_onnx_ms:.2f}x':>20s}")
    
    print(f"\n  PyTorch CPU vs ONNX CPU:")
    print(f"    PyTorch CPU: {bert_pytorch_ms:.2f} ms")
    print(f"    ONNX CPU:    {bert_onnx_ms:.2f} ms")
    print(f"    ONNX speedup: {bert_pytorch_ms/bert_onnx_ms:.2f}x")
    
    # Save results
    results = {
        "TF-IDF + LogReg": {"cpu_inference_ms": round(baseline_ms, 2)},
        "BiLSTM (PyTorch)": {"cpu_inference_ms": round(bilstm_ms, 2)},
        "DistilBERT (PyTorch)": {"cpu_inference_ms": round(bert_pytorch_ms, 2)},
        "DistilBERT (ONNX)": {"cpu_inference_ms": round(bert_onnx_ms, 2)},
        "onnx_speedup_vs_pytorch_cpu": round(bert_pytorch_ms / bert_onnx_ms, 2),
    }
    
    with open("models/cpu_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: models/cpu_benchmark.json")
    print(f"\n  KEY INSIGHT: ONNX provides ~{bert_pytorch_ms/bert_onnx_ms:.1f}x speedup over PyTorch on CPU")
    print(f"               (this is the FAIR comparison — same hardware)")