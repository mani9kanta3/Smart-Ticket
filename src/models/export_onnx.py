"""
SmartTicket - ONNX Export & Model Comparison
Exports DistilBERT to ONNX format for faster inference,
then benchmarks all models side by side.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, classification_report
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# ============================================================
# Re-define the model class (needed to load weights)
# ============================================================

class DistilBertMultiTask(nn.Module):
    """Same architecture as in bert_train.py"""
    
    def __init__(self, num_categories=10, num_priorities=4, dropout=0.3):
        super(DistilBertMultiTask, self).__init__()
        self.bert = DistilBertModel.from_pretrained("models/bert_finetuned")
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_categories),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_priorities),
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        category_logits = self.category_head(cls_output)
        priority_logits = self.priority_head(cls_output)
        return category_logits, priority_logits


# ============================================================
# ONNX Export
# ============================================================

def export_to_onnx():
    """
    Export the fine-tuned DistilBERT to ONNX format.
    
    ONNX = Open Neural Network Exchange
    - Platform-independent model format
    - ONNX Runtime applies graph optimizations automatically
    - Runs without PyTorch installed (lighter deployment)
    """
    print("=" * 60)
    print("STEP 1: Export DistilBERT to ONNX")
    print("=" * 60)
    
    # Load the fine-tuned model
    model = DistilBertMultiTask(num_categories=10, num_priorities=4)
    checkpoint = torch.load("models/bert_finetuned/full_model.pth",
                            map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("  Model loaded successfully")
    
    # Create dummy input (ONNX needs example input to trace the model)
    max_length = 64
    dummy_input_ids = torch.randint(0, 1000, (1, max_length))
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    
    # Export to ONNX
    onnx_path = "models/smartticket_bert.onnx"
    
    print("  Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["category_logits", "priority_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "category_logits": {0: "batch_size"},
            "priority_logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    # Check file size
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX model saved: {onnx_path}")
    print(f"  ONNX model size: {onnx_size_mb:.1f} MB")
    
    # Verify ONNX model loads correctly
    session = ort.InferenceSession(onnx_path)
    print(f"  ONNX Runtime verification: ✅ Model loads successfully")
    print(f"  ONNX Runtime providers: {session.get_providers()}")
    
    return onnx_path, onnx_size_mb


# ============================================================
# Accuracy Verification
# ============================================================

def verify_onnx_accuracy(onnx_path):
    """
    Verify that ONNX model produces the same predictions as PyTorch.
    If accuracy drops after export, something went wrong.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Verify ONNX Accuracy")
    print("=" * 60)
    
    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    tokenizer = DistilBertTokenizer.from_pretrained("models/bert_finetuned")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    
    # Load PyTorch model for comparison
    pytorch_model = DistilBertMultiTask(num_categories=10, num_priorities=4)
    checkpoint = torch.load("models/bert_finetuned/full_model.pth",
                            map_location="cpu", weights_only=True)
    pytorch_model.load_state_dict(checkpoint["model_state_dict"])
    pytorch_model.eval()
    
    max_length = 64
    onnx_cat_preds = []
    onnx_pri_preds = []
    pytorch_cat_preds = []
    pytorch_pri_preds = []
    cat_labels = test_df["category_id"].tolist()
    pri_labels = test_df["priority_id"].tolist()
    
    print("  Running predictions on test set...")
    
    for i, text in enumerate(test_df["text"]):
        # Tokenize
        encoding = tokenizer(
            text, max_length=max_length,
            padding="max_length", truncation=True,
            return_tensors="np",
        )
        
        input_ids_np = encoding["input_ids"].astype(np.int64)
        attention_mask_np = encoding["attention_mask"].astype(np.int64)
        
        # ONNX prediction
        onnx_out = session.run(
            None,
            {"input_ids": input_ids_np, "attention_mask": attention_mask_np},
        )
        onnx_cat_preds.append(np.argmax(onnx_out[0], axis=1)[0])
        onnx_pri_preds.append(np.argmax(onnx_out[1], axis=1)[0])
        
        # PyTorch prediction
        with torch.no_grad():
            input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)
            attention_mask_pt = torch.tensor(attention_mask_np, dtype=torch.long)
            pt_cat, pt_pri = pytorch_model(input_ids_pt, attention_mask_pt)
            pytorch_cat_preds.append(pt_cat.argmax(dim=1).item())
            pytorch_pri_preds.append(pt_pri.argmax(dim=1).item())
        
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(test_df)}")
    
    # Compare results
    onnx_cat_acc = accuracy_score(cat_labels, onnx_cat_preds)
    onnx_cat_f1 = f1_score(cat_labels, onnx_cat_preds, average="weighted")
    pytorch_cat_acc = accuracy_score(cat_labels, pytorch_cat_preds)
    pytorch_cat_f1 = f1_score(cat_labels, pytorch_cat_preds, average="weighted")
    
    onnx_pri_acc = accuracy_score(pri_labels, onnx_pri_preds)
    onnx_pri_f1 = f1_score(pri_labels, onnx_pri_preds, average="weighted")
    
    # Check if predictions match
    match_count = sum(a == b for a, b in zip(onnx_cat_preds, pytorch_cat_preds))
    match_pct = match_count / len(onnx_cat_preds) * 100
    
    print(f"\n  PyTorch vs ONNX prediction match: {match_count}/{len(onnx_cat_preds)} ({match_pct:.1f}%)")
    print(f"\n  {'Metric':<20s} {'PyTorch':>10s} {'ONNX':>10s} {'Diff':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'Cat Accuracy':<20s} {pytorch_cat_acc:>10.4f} {onnx_cat_acc:>10.4f} {onnx_cat_acc-pytorch_cat_acc:>+10.4f}")
    print(f"  {'Cat F1':<20s} {pytorch_cat_f1:>10.4f} {onnx_cat_f1:>10.4f} {onnx_cat_f1-pytorch_cat_f1:>+10.4f}")
    print(f"  {'Pri Accuracy':<20s} {'-':>10s} {onnx_pri_acc:>10.4f} {'-':>10s}")
    print(f"  {'Pri F1':<20s} {'-':>10s} {onnx_pri_f1:>10.4f} {'-':>10s}")
    
    return onnx_cat_acc, onnx_cat_f1, onnx_pri_acc, onnx_pri_f1


# ============================================================
# Inference Speed Benchmark
# ============================================================

def benchmark_speed(onnx_path):
    """
    Compare inference speed: PyTorch vs ONNX Runtime.
    This is the key production metric.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Inference Speed Benchmark")
    print("=" * 60)
    
    tokenizer = DistilBertTokenizer.from_pretrained("models/bert_finetuned")
    max_length = 64
    
    # Sample input
    sample_text = "my card payment was declined and I need help urgently"
    encoding = tokenizer(
        sample_text, max_length=max_length,
        padding="max_length", truncation=True,
    )
    
    num_runs = 200
    
    # --- PyTorch CPU benchmark ---
    pytorch_model = DistilBertMultiTask(num_categories=10, num_priorities=4)
    checkpoint = torch.load("models/bert_finetuned/full_model.pth",
                            map_location="cpu", weights_only=True)
    pytorch_model.load_state_dict(checkpoint["model_state_dict"])
    pytorch_model.eval()
    
    input_ids_pt = torch.tensor([encoding["input_ids"]], dtype=torch.long)
    attention_mask_pt = torch.tensor([encoding["attention_mask"]], dtype=torch.long)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            pytorch_model(input_ids_pt, attention_mask_pt)
    
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            pytorch_model(input_ids_pt, attention_mask_pt)
    pytorch_cpu_ms = (time.time() - start) / num_runs * 1000
    
    # --- PyTorch GPU benchmark ---
    pytorch_gpu_ms = None
    if torch.cuda.is_available():
        pytorch_model_gpu = pytorch_model.to(DEVICE)
        input_ids_gpu = input_ids_pt.to(DEVICE)
        attention_mask_gpu = attention_mask_pt.to(DEVICE)
        
        for _ in range(20):
            with torch.no_grad():
                pytorch_model_gpu(input_ids_gpu, attention_mask_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                pytorch_model_gpu(input_ids_gpu, attention_mask_gpu)
        torch.cuda.synchronize()
        pytorch_gpu_ms = (time.time() - start) / num_runs * 1000
        
        # Move back to CPU to free GPU memory
        pytorch_model_gpu.cpu()
        del pytorch_model_gpu, input_ids_gpu, attention_mask_gpu
        torch.cuda.empty_cache()
    
    # --- ONNX Runtime CPU benchmark ---
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    input_ids_np = np.array([encoding["input_ids"]], dtype=np.int64)
    attention_mask_np = np.array([encoding["attention_mask"]], dtype=np.int64)
    
    for _ in range(20):
        session.run(None, {"input_ids": input_ids_np, "attention_mask": attention_mask_np})
    
    start = time.time()
    for _ in range(num_runs):
        session.run(None, {"input_ids": input_ids_np, "attention_mask": attention_mask_np})
    onnx_cpu_ms = (time.time() - start) / num_runs * 1000
    
    # Results
    print(f"\n  Inference Speed (single query, {num_runs} runs):")
    print(f"  {'Runtime':<25s} {'Time (ms)':>10s} {'Speedup':>10s}")
    print(f"  {'-'*45}")
    print(f"  {'PyTorch CPU':<25s} {pytorch_cpu_ms:>10.2f} {'1.0x':>10s}")
    if pytorch_gpu_ms:
        speedup_gpu = pytorch_cpu_ms / pytorch_gpu_ms
        print(f"  {'PyTorch GPU':<25s} {pytorch_gpu_ms:>10.2f} {f'{speedup_gpu:.1f}x':>10s}")
    speedup_onnx = pytorch_cpu_ms / onnx_cpu_ms
    print(f"  {'ONNX Runtime CPU':<25s} {onnx_cpu_ms:>10.2f} {f'{speedup_onnx:.1f}x':>10s}")
    
    return pytorch_cpu_ms, pytorch_gpu_ms, onnx_cpu_ms


# ============================================================
# Comprehensive Model Comparison
# ============================================================

def create_comparison_report(onnx_cat_acc, onnx_cat_f1, onnx_pri_acc, onnx_pri_f1,
                             pytorch_cpu_ms, pytorch_gpu_ms, onnx_cpu_ms, onnx_size_mb):
    """Create the final 4-model comparison with visualizations."""
    print("\n" + "=" * 60)
    print("STEP 4: Comprehensive Model Comparison")
    print("=" * 60)
    
    # All model results
    models_data = {
        "Model": [
            "TF-IDF + LogReg",
            "BiLSTM (PyTorch)",
            "DistilBERT (PyTorch)",
            "DistilBERT (ONNX)",
        ],
        "Cat Accuracy": [0.8753, 0.8781, 0.9058, onnx_cat_acc],
        "Cat F1": [0.8752, 0.8780, 0.9058, onnx_cat_f1],
        "Pri Accuracy": [0.8927, 0.9051, 0.9155, onnx_pri_acc],
        "Pri F1": [0.8928, 0.9050, 0.9155, onnx_pri_f1],
        "Inference (ms)": [0.39, 14.15, pytorch_gpu_ms or pytorch_cpu_ms, onnx_cpu_ms],
        "Model Size (MB)": [5, 50, 250, onnx_size_mb],
    }
    
    df = pd.DataFrame(models_data)
    
    print(f"\n  {'Model':<25s} {'Cat Acc':>9s} {'Cat F1':>9s} {'Pri Acc':>9s} {'Inf(ms)':>9s} {'Size(MB)':>9s}")
    print(f"  {'-'*70}")
    for _, row in df.iterrows():
        print(f"  {row['Model']:<25s} {row['Cat Accuracy']:>8.1%} {row['Cat F1']:>9.4f} {row['Pri Accuracy']:>8.1%} {row['Inference (ms)']:>9.2f} {row['Model Size (MB)']:>8.1f}")
    
    # ----------------------------------------------------------
    # Chart 1: Accuracy Comparison
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = range(len(df))
    colors = ["#2196F3", "#4CAF50", "#9C27B0", "#FF5722"]
    
    bars1 = axes[0].bar(x, df["Cat Accuracy"], color=colors, width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Model"], rotation=20, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Category Classification Accuracy")
    axes[0].set_ylim(0.8, 0.95)
    for bar, val in zip(bars1, df["Cat Accuracy"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{val:.1%}", ha="center", fontweight="bold", fontsize=10)
    
    bars2 = axes[1].bar(x, df["Cat F1"], color=colors, width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["Model"], rotation=20, ha="right")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Category F1 Score")
    axes[1].set_ylim(0.8, 0.95)
    for bar, val in zip(bars2, df["Cat F1"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{val:.4f}", ha="center", fontweight="bold", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("reports/figures/model_accuracy_comparison.png", dpi=150)
    plt.close()
    print("\n  Saved: reports/figures/model_accuracy_comparison.png")
    
    # ----------------------------------------------------------
    # Chart 2: Inference Speed Comparison
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df["Model"], df["Inference (ms)"], color=colors, width=0.6)
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Speed Comparison (lower is better)")
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    for bar, val in zip(bars, df["Inference (ms)"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}ms", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/inference_speed_comparison.png", dpi=150)
    plt.close()
    print("  Saved: reports/figures/inference_speed_comparison.png")
    
    # ----------------------------------------------------------
    # Chart 3: Accuracy vs Speed Tradeoff
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in df.iterrows():
        ax.scatter(row["Inference (ms)"], row["Cat F1"],
                   s=row["Model Size (MB)"] * 2 + 50,
                   c=colors[i], label=row["Model"], zorder=5, edgecolors="black")
    ax.set_xlabel("Inference Time (ms) — lower is better →")
    ax.set_ylabel("F1 Score — higher is better →")
    ax.set_title("Accuracy vs Speed Tradeoff (bubble size = model size)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/accuracy_vs_speed_tradeoff.png", dpi=150)
    plt.close()
    print("  Saved: reports/figures/accuracy_vs_speed_tradeoff.png")
    
    # Save comparison data as JSON
    comparison = {row["Model"]: {
        "cat_accuracy": row["Cat Accuracy"],
        "cat_f1": row["Cat F1"],
        "pri_accuracy": row["Pri Accuracy"],
        "pri_f1": row["Pri F1"],
        "inference_ms": row["Inference (ms)"],
        "model_size_mb": row["Model Size (MB)"],
    } for _, row in df.iterrows()}
    
    with open("models/model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("  Saved: models/model_comparison.json")
    
    return df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("SmartTicket - ONNX Export & Model Comparison")
    print("=" * 60)
    
    # Step 1: Export to ONNX
    onnx_path, onnx_size_mb = export_to_onnx()
    
    # Step 2: Verify accuracy
    onnx_cat_acc, onnx_cat_f1, onnx_pri_acc, onnx_pri_f1 = verify_onnx_accuracy(onnx_path)
    
    # Step 3: Benchmark speed
    pytorch_cpu_ms, pytorch_gpu_ms, onnx_cpu_ms = benchmark_speed(onnx_path)
    
    # Step 4: Full comparison
    comparison_df = create_comparison_report(
        onnx_cat_acc, onnx_cat_f1, onnx_pri_acc, onnx_pri_f1,
        pytorch_cpu_ms, pytorch_gpu_ms, onnx_cpu_ms, onnx_size_mb,
    )
    
    print("\n" + "=" * 60)
    print("ONNX EXPORT & COMPARISON COMPLETE!")
    print("=" * 60)
    print("\n  Key takeaway: DistilBERT + ONNX gives us the BEST accuracy")
    print("  AND fast inference — the best of both worlds for production.")