"""
SmartTicket - DistilBERT Fine-tuning
Fine-tunes a pre-trained DistilBERT model for multi-task
ticket classification (category + priority).

Key difference from BiLSTM:
- BiLSTM learned language from scratch on 11K samples
- DistilBERT was pre-trained on BILLIONS of words
- We just adapt it to our specific task (transfer learning)
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import mlflow
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Check available GPU memory
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {total_mem:.1f} GB")

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models/bert_finetuned", exist_ok=True)


# ============================================================
# PART 1: Dataset for BERT
# ============================================================

class BertTicketDataset(Dataset):
    """
    Dataset that tokenizes text using DistilBERT's tokenizer.
    
    Unlike our BiLSTM dataset (which just split on spaces),
    BERT uses subword tokenization:
      "unrecognized" → ["un", "##rec", "##og", "##nized"]
    
    This means BERT can handle words it's never seen before
    by breaking them into familiar subword pieces.
    """
    
    def __init__(self, texts, category_labels, priority_labels, tokenizer, max_length=64):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.category_labels = category_labels.tolist() if hasattr(category_labels, 'tolist') else list(category_labels)
        self.priority_labels = priority_labels.tolist() if hasattr(priority_labels, 'tolist') else list(priority_labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize with BERT tokenizer
        # Returns: input_ids (token indices), attention_mask (1 for real tokens, 0 for padding)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "category_label": torch.tensor(self.category_labels[idx], dtype=torch.long),
            "priority_label": torch.tensor(self.priority_labels[idx], dtype=torch.long),
        }


# ============================================================
# PART 2: Multi-Task DistilBERT Model
# ============================================================

class DistilBertMultiTask(nn.Module):
    """
    DistilBERT with two classification heads.
    
    Architecture:
      Input text → DistilBERT Tokenizer → Token IDs
          ↓
      DistilBERT (6 transformer layers, pre-trained)
          ↓
      [CLS] token representation (768 dimensions)
          ↓
      Dropout (0.3)
          ↓
      ┌─────────────────────────────────┐
      │ Category Head: 768 → 256 → 10   │
      │ Priority Head: 768 → 128 → 4    │
      └─────────────────────────────────┘
    
    We freeze the first 4 transformer layers and only train
    the last 2 layers + classification heads. This:
    - Preserves general language understanding (layers 1-4)
    - Adapts higher-level patterns to our task (layers 5-6)
    - Prevents catastrophic forgetting
    - Reduces training time and memory usage
    """
    
    def __init__(self, num_categories=10, num_priorities=4, dropout=0.3):
        super(DistilBertMultiTask, self).__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        # Freeze first 4 of 6 transformer layers
        # Layer 0-3: frozen (general language knowledge)
        # Layer 4-5: trainable (adapt to our task)
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.bert.transformer.layer):
            if i < 4:
                for param in layer.parameters():
                    param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size  # 768
        
        self.dropout = nn.Dropout(dropout)
        
        # Category classification head
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_categories),
        )
        
        # Priority classification head
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_priorities),
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_length) — token indices
            attention_mask: (batch_size, seq_length) — 1 for real tokens, 0 for padding
        
        Returns:
            category_logits, priority_logits
        """
        # Get DistilBERT output
        # last_hidden_state: (batch_size, seq_length, 768)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        # This token is trained to capture the overall meaning of the input
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        cls_output = self.dropout(cls_output)
        
        category_logits = self.category_head(cls_output)
        priority_logits = self.priority_head(cls_output)
        
        return category_logits, priority_logits


# ============================================================
# PART 3: Training Functions
# ============================================================

def train_one_epoch(model, dataloader, cat_criterion, pri_criterion,
                    optimizer, scheduler, device, accumulation_steps=2):
    """
    Train for one epoch with gradient accumulation.
    
    Gradient accumulation: with batch_size=8 and accumulation_steps=2,
    we effectively train with batch_size=16 but only use memory for 8.
    This lets us train larger effective batches on limited GPU memory.
    """
    model.train()
    total_loss = 0
    all_cat_preds, all_cat_labels = [], []
    all_pri_preds, all_pri_labels = [], []
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cat_labels = batch["category_label"].to(device)
        pri_labels = batch["priority_label"].to(device)
        
        # Forward pass
        cat_logits, pri_logits = model(input_ids, attention_mask)
        
        # Combined loss
        cat_loss = cat_criterion(cat_logits, cat_labels)
        pri_loss = pri_criterion(pri_logits, pri_labels)
        loss = (cat_loss + pri_loss) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        all_cat_preds.extend(cat_logits.argmax(dim=1).cpu().numpy())
        all_cat_labels.extend(cat_labels.cpu().numpy())
        all_pri_preds.extend(pri_logits.argmax(dim=1).cpu().numpy())
        all_pri_labels.extend(pri_labels.cpu().numpy())
        
        # Free GPU memory
        del input_ids, attention_mask, cat_labels, pri_labels, cat_logits, pri_logits
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    cat_acc = accuracy_score(all_cat_labels, all_cat_preds)
    cat_f1 = f1_score(all_cat_labels, all_cat_preds, average="weighted")
    pri_acc = accuracy_score(all_pri_labels, all_pri_preds)
    
    return avg_loss, cat_acc, cat_f1, pri_acc


def evaluate(model, dataloader, cat_criterion, pri_criterion, device):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0
    all_cat_preds, all_cat_labels = [], []
    all_pri_preds, all_pri_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cat_labels = batch["category_label"].to(device)
            pri_labels = batch["priority_label"].to(device)
            
            cat_logits, pri_logits = model(input_ids, attention_mask)
            
            cat_loss = cat_criterion(cat_logits, cat_labels)
            pri_loss = pri_criterion(pri_logits, pri_labels)
            loss = cat_loss + pri_loss
            
            total_loss += loss.item()
            
            all_cat_preds.extend(cat_logits.argmax(dim=1).cpu().numpy())
            all_cat_labels.extend(cat_labels.cpu().numpy())
            all_pri_preds.extend(pri_logits.argmax(dim=1).cpu().numpy())
            all_pri_labels.extend(pri_labels.cpu().numpy())
            
            del input_ids, attention_mask, cat_labels, pri_labels
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataloader)
    cat_acc = accuracy_score(all_cat_labels, all_cat_preds)
    cat_f1 = f1_score(all_cat_labels, all_cat_preds, average="weighted")
    pri_acc = accuracy_score(all_pri_labels, all_pri_preds)
    pri_f1 = f1_score(all_pri_labels, all_pri_preds, average="weighted")
    
    return (avg_loss, cat_acc, cat_f1, pri_acc, pri_f1,
            all_cat_preds, all_cat_labels, all_pri_preds, all_pri_labels)


def compute_class_weights(labels, num_classes, device):
    """Compute balanced class weights."""
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights).to(device)


# ============================================================
# PART 4: Main Training Pipeline
# ============================================================

if __name__ == "__main__":
    print("SmartTicket - DistilBERT Fine-tuning")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # Load Data
    # ----------------------------------------------------------
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # ----------------------------------------------------------
    # Tokenizer & Datasets
    # ----------------------------------------------------------
    print("\nLoading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Use smaller max_length to save GPU memory (our queries avg 11 words)
    max_length = 64
    batch_size = 8  # Small batch for 4GB GPU
    accumulation_steps = 2  # Effective batch size = 8 * 2 = 16
    
    train_dataset = BertTicketDataset(
        train_df["text"], train_df["category_id"], train_df["priority_id"],
        tokenizer, max_length=max_length,
    )
    val_dataset = BertTicketDataset(
        val_df["text"], val_df["category_id"], val_df["priority_id"],
        tokenizer, max_length=max_length,
    )
    test_dataset = BertTicketDataset(
        test_df["text"], test_df["category_id"], test_df["priority_id"],
        tokenizer, max_length=max_length,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Max length: {max_length}")
    print(f"  Batch size: {batch_size} (effective: {batch_size * accumulation_steps})")
    print(f"  Train batches: {len(train_loader)}")
    
    # ----------------------------------------------------------
    # Build Model
    # ----------------------------------------------------------
    print("\nBuilding DistilBERT multi-task model...")
    model = DistilBertMultiTask(num_categories=10, num_priorities=4).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    # ----------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # ----------------------------------------------------------
    cat_weights = compute_class_weights(train_df["category_id"].values, 10, DEVICE)
    pri_weights = compute_class_weights(train_df["priority_id"].values, 4, DEVICE)
    
    cat_criterion = nn.CrossEntropyLoss(weight=cat_weights)
    pri_criterion = nn.CrossEntropyLoss(weight=pri_weights)
    
    # AdamW: Adam with proper weight decay (standard for BERT fine-tuning)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,  # Very small learning rate — don't destroy BERT's knowledge
        weight_decay=0.01,
    )
    
    # Learning rate warmup then linear decay
    epochs = 5
    total_steps = len(train_loader) * epochs // accumulation_steps
    warmup_steps = total_steps // 10  # Warmup for first 10% of training
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\n  Epochs: {epochs}")
    print(f"  Learning rate: 2e-5")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # ----------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting DistilBERT fine-tuning...")
    print("=" * 60)
    
    best_val_f1 = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_cat_f1": [], "val_cat_f1": [],
    }
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name="distilbert_multitask"):
        
        mlflow.log_param("model_type", "DistilBERT Fine-tuned")
        mlflow.log_param("base_model", "distilbert-base-uncased")
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("effective_batch_size", batch_size * accumulation_steps)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("frozen_layers", 4)
        mlflow.log_param("trainable_params", trainable_params)
        
        training_start = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            train_loss, train_cat_acc, train_cat_f1, train_pri_acc = train_one_epoch(
                model, train_loader, cat_criterion, pri_criterion,
                optimizer, scheduler, DEVICE, accumulation_steps,
            )
            
            val_loss, val_cat_acc, val_cat_f1, val_pri_acc, val_pri_f1, _, _, _, _ = evaluate(
                model, val_loader, cat_criterion, pri_criterion, DEVICE,
            )
            
            epoch_time = time.time() - epoch_start
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_cat_f1"].append(train_cat_f1)
            history["val_cat_f1"].append(val_cat_f1)
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_cat_f1", val_cat_f1, step=epoch)
            
            print(f"\n  Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"    Train Loss: {train_loss:.4f} | Cat Acc: {train_cat_acc:.4f} | Cat F1: {train_cat_f1:.4f}")
            print(f"    Val   Loss: {val_loss:.4f} | Cat Acc: {val_cat_acc:.4f} | Cat F1: {val_cat_f1:.4f} | Pri Acc: {val_pri_acc:.4f}")
            
            if val_cat_f1 > best_val_f1:
                best_val_f1 = val_cat_f1
                # Save model and tokenizer
                model.bert.save_pretrained("models/bert_finetuned")
                tokenizer.save_pretrained("models/bert_finetuned")
                # Save full model state (including classification heads)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "num_categories": 10,
                        "num_priorities": 4,
                        "max_length": max_length,
                    },
                }, "models/bert_finetuned/full_model.pth")
                print(f"    ✅ New best model saved! (Val F1: {best_val_f1:.4f})")
            else:
                print(f"    ⏳ No improvement")
        
        total_train_time = time.time() - training_start
        print(f"\n  Total training time: {total_train_time:.1f} seconds")
        
        # ----------------------------------------------------------
        # Final Test Evaluation
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        
        # Load best model
        checkpoint = torch.load("models/bert_finetuned/full_model.pth",
                                map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        test_loss, test_cat_acc, test_cat_f1, test_pri_acc, test_pri_f1, \
            cat_preds, cat_labels, pri_preds, pri_labels = evaluate(
                model, test_loader, cat_criterion, pri_criterion, DEVICE,
            )
        
        id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
        category_names = [id_to_cat[i] for i in range(10)]
        id_to_pri = {int(k): v for k, v in mappings["id_to_priority"].items()}
        priority_names = [id_to_pri[i] for i in range(4)]
        
        print(f"\n  Category Classification:")
        print(f"    Test Accuracy: {test_cat_acc:.4f}")
        print(f"    Test F1 Score: {test_cat_f1:.4f}")
        print(f"\n{classification_report(cat_labels, cat_preds, target_names=category_names)}")
        
        print(f"\n  Priority Classification:")
        print(f"    Test Accuracy: {test_pri_acc:.4f}")
        print(f"    Test F1 Score: {test_pri_f1:.4f}")
        print(f"\n{classification_report(pri_labels, pri_preds, target_names=priority_names)}")
        
        # Log final metrics
        mlflow.log_metric("test_cat_accuracy", test_cat_acc)
        mlflow.log_metric("test_cat_f1", test_cat_f1)
        mlflow.log_metric("test_pri_accuracy", test_pri_acc)
        mlflow.log_metric("test_pri_f1", test_pri_f1)
        mlflow.log_metric("training_time_sec", total_train_time)
        
        # ----------------------------------------------------------
        # Confusion Matrix
        # ----------------------------------------------------------
        cm = confusion_matrix(cat_labels, cat_preds)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                    xticklabels=category_names, yticklabels=category_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"DistilBERT Category Confusion Matrix (Test Accuracy: {test_cat_acc:.1%})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("reports/figures/bert_category_confusion_matrix.png", dpi=150)
        plt.close()
        mlflow.log_artifact("reports/figures/bert_category_confusion_matrix.png")
        
        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        
        axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("DistilBERT Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(epochs_range, history["train_cat_f1"], label="Train F1", marker="o")
        axes[1].plot(epochs_range, history["val_cat_f1"], label="Val F1", marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("DistilBERT Training & Validation F1")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig("reports/figures/bert_training_curves.png", dpi=150)
        plt.close()
        mlflow.log_artifact("reports/figures/bert_training_curves.png")
        
        # ----------------------------------------------------------
        # Inference Speed
        # ----------------------------------------------------------
        model.eval()
        sample = tokenizer(
            "my payment failed",
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sample_ids = sample["input_ids"].to(DEVICE)
        sample_mask = sample["attention_mask"].to(DEVICE)
        
        for _ in range(10):
            with torch.no_grad():
                model(sample_ids, sample_mask)
        
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                model(sample_ids, sample_mask)
        avg_ms = (time.time() - start) / 100 * 1000
        mlflow.log_metric("inference_ms", avg_ms)
        
        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("DistilBERT TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\n  Category Model:")
        print(f"    Test Accuracy: {test_cat_acc:.1%}")
        print(f"    Test F1 Score: {test_cat_f1:.4f}")
        print(f"\n  Priority Model:")
        print(f"    Test Accuracy: {test_pri_acc:.1%}")
        print(f"    Test F1 Score: {test_pri_f1:.4f}")
        print(f"\n  Inference: {avg_ms:.2f} ms/query")
        print(f"  Training time: {total_train_time:.1f}s")
        print(f"\n  Model Comparison:")
        print(f"    {'Model':<25s} {'Accuracy':>10s} {'F1':>10s}")
        print(f"    {'-'*45}")
        print(f"    {'TF-IDF + LogReg':<25s} {'87.5%':>10s} {'0.8752':>10s}")
        print(f"    {'BiLSTM':<25s} {'87.8%':>10s} {'0.8780':>10s}")
        print(f"    {'DistilBERT':<25s} {f'{test_cat_acc:.1%}':>10s} {f'{test_cat_f1:.4f}':>10s}")