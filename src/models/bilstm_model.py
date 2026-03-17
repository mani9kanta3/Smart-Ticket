"""
SmartTicket - BiLSTM Model (PyTorch from Scratch)
Custom Dataset, BiLSTM architecture, and training loop.

Architecture:
  Text → Vocabulary Lookup → Embedding (300d) → BiLSTM (2 layers, 256 hidden)
  → Dropout → Category Head (10 classes) + Priority Head (4 classes)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import mlflow
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# PART 1: Custom PyTorch Dataset
# ============================================================

class TicketDataset(Dataset):
    """
    Custom Dataset that converts text queries into number sequences.
    
    How it works:
    1. Build vocabulary from training text: each unique word gets a number
       "payment" → 45, "failed" → 128, "card" → 67
    2. Convert each query to a sequence of numbers:
       "my payment failed" → [12, 45, 128]
    3. Pad sequences to same length (PyTorch batches need same-size tensors):
       "card stolen" → [67, 203, 0, 0, 0, ...] (padded with 0s)
    """
    
    def __init__(self, texts, category_labels, priority_labels,
                 vocab=None, max_length=64, min_freq=2):
        """
        Args:
            texts: list of query strings
            category_labels: list of category IDs (0-9)
            priority_labels: list of priority IDs (0-3)
            vocab: existing vocabulary dict (pass None for training set to build new)
            max_length: maximum sequence length (queries longer than this get trimmed)
            min_freq: minimum word frequency to include in vocabulary
        """
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.category_labels = category_labels.tolist() if hasattr(category_labels, 'tolist') else list(category_labels)
        self.priority_labels = priority_labels.tolist() if hasattr(priority_labels, 'tolist') else list(priority_labels)
        self.max_length = max_length
        
        # Build or use existing vocabulary
        if vocab is None:
            self.vocab = self._build_vocab(min_freq)
            print(f"  Built vocabulary: {len(self.vocab)} words (min_freq={min_freq})")
        else:
            self.vocab = vocab
    
    def _build_vocab(self, min_freq):
        """
        Build vocabulary from text. Only include words that appear
        at least min_freq times (filters out typos and rare words).
        
        Special tokens:
          <PAD> = 0: padding token (fills short sequences)
          <UNK> = 1: unknown token (replaces words not in vocabulary)
        """
        word_counts = Counter()
        for text in self.texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for word, count in word_counts.most_common():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
        
        return vocab
    
    def _text_to_sequence(self, text):
        """
        Convert text string to list of vocabulary indices.
        Words not in vocabulary get mapped to <UNK> (index 1).
        """
        words = text.lower().split()
        sequence = [self.vocab.get(w, self.vocab["<UNK>"]) for w in words]
        
        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # Pad if too short
        padding_length = self.max_length - len(sequence)
        sequence = sequence + [self.vocab["<PAD>"]] * padding_length
        
        return sequence
    
    def __len__(self):
        """PyTorch needs this to know dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        PyTorch calls this to get one sample.
        Returns: (text_tensor, category_label, priority_label)
        """
        sequence = self._text_to_sequence(self.texts[idx])
        
        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(self.category_labels[idx], dtype=torch.long),
            torch.tensor(self.priority_labels[idx], dtype=torch.long),
        )


# ============================================================
# PART 2: BiLSTM Architecture
# ============================================================

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for multi-task text classification.
    
    Architecture diagram:
    
    Input: "my payment failed" → [12, 45, 128]
              ↓
    ┌─────────────────────────┐
    │  Embedding Layer         │  Each word index → 300-dim vector
    │  (vocab_size × 300)      │  These vectors are LEARNED during training
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  BiLSTM                  │  Reads sequence left→right AND right→left
    │  (2 layers, 256 hidden)  │  Captures word ORDER and CONTEXT
    │  + Dropout 0.3           │  Output: 512-dim (256 forward + 256 backward)
    └─────────────────────────┘
              ↓
    ┌─────────────────────────┐
    │  Dropout 0.5             │  Prevents overfitting
    └─────────────────────────┘
              ↓
    ┌──────────────────────────────────────────┐
    │  Category Head: Linear(512 → 10)          │  → Predicts category
    │  Priority Head: Linear(512 → 4)           │  → Predicts priority
    └──────────────────────────────────────────┘
    """
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256,
                 num_layers=2, num_categories=10, num_priorities=4,
                 dropout=0.3, fc_dropout=0.5, pad_idx=0):
        super(BiLSTMClassifier, self).__init__()
        
        # Embedding: word index → dense vector
        # padding_idx=0 means the <PAD> token always maps to zero vector
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        
        # Bidirectional LSTM
        # bidirectional=True means two LSTMs run in parallel:
        #   Forward LSTM:  reads "my payment failed" left→right
        #   Backward LSTM: reads "my payment failed" right→left
        # Their outputs are concatenated → 256*2 = 512 dimensions
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Dropout: randomly zeros out neurons during training
        # Forces model to not rely on any single neuron → more robust
        self.dropout = nn.Dropout(fc_dropout)
        
        # Two separate classification heads (multi-task learning)
        # Both share the same LSTM body but make independent predictions
        self.category_head = nn.Linear(hidden_dim * 2, num_categories)
        self.priority_head = nn.Linear(hidden_dim * 2, num_priorities)
    
    def forward(self, x):
        """
        Forward pass: input text indices → category + priority predictions
        
        Args:
            x: tensor of shape (batch_size, seq_length) containing word indices
        
        Returns:
            category_logits: (batch_size, 10) — raw scores for each category
            priority_logits: (batch_size, 4) — raw scores for each priority
        """
        # Step 1: Embedding lookup
        # (batch_size, seq_length) → (batch_size, seq_length, 300)
        embedded = self.embedding(x)
        
        # Step 2: BiLSTM processes the sequence
        # lstm_out: (batch_size, seq_length, 512) — output at every time step
        # (hidden, cell): final hidden and cell states
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Step 3: Get the final representation
        # Concatenate the last hidden state from forward and backward LSTMs
        # hidden shape: (num_layers*2, batch_size, hidden_dim)
        # We take the last layer: forward [-2] and backward [-1]
        forward_hidden = hidden[-2]  # (batch_size, 256)
        backward_hidden = hidden[-1]  # (batch_size, 256)
        combined = torch.cat((forward_hidden, backward_hidden), dim=1)  # (batch_size, 512)
        
        # Step 4: Dropout for regularization
        combined = self.dropout(combined)
        
        # Step 5: Classification heads
        category_logits = self.category_head(combined)
        priority_logits = self.priority_head(combined)
        
        return category_logits, priority_logits
    
# ============================================================
# PART 3: Training & Evaluation Functions
# ============================================================

def train_one_epoch(model, dataloader, cat_criterion, pri_criterion,
                    optimizer, device):
    """
    One pass through all training data.
    For each batch:
      1. Forward pass → get predictions
      2. Calculate loss (how wrong are predictions?)
      3. Backward pass → calculate gradients
      4. Update weights → move in direction that reduces loss
    """
    model.train()  # Enable dropout (only active during training)
    total_loss = 0
    all_cat_preds, all_cat_labels = [], []
    all_pri_preds, all_pri_labels = [], []
    
    for batch_idx, (texts, cat_labels, pri_labels) in enumerate(dataloader):
        texts = texts.to(device)
        cat_labels = cat_labels.to(device)
        pri_labels = pri_labels.to(device)
        
        # Forward pass
        cat_logits, pri_logits = model(texts)
        
        # Calculate combined loss (category + priority)
        # We weight them equally, but you could adjust this
        cat_loss = cat_criterion(cat_logits, cat_labels)
        pri_loss = pri_criterion(pri_logits, pri_labels)
        loss = cat_loss + pri_loss
        
        # Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Calculate new gradients
        
        # Gradient clipping: prevents "exploding gradients"
        # where weights update too dramatically and training becomes unstable
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
        
        # Collect predictions for metrics
        all_cat_preds.extend(cat_logits.argmax(dim=1).cpu().numpy())
        all_cat_labels.extend(cat_labels.cpu().numpy())
        all_pri_preds.extend(pri_logits.argmax(dim=1).cpu().numpy())
        all_pri_labels.extend(pri_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    cat_acc = accuracy_score(all_cat_labels, all_cat_preds)
    cat_f1 = f1_score(all_cat_labels, all_cat_preds, average="weighted")
    pri_acc = accuracy_score(all_pri_labels, all_pri_preds)
    
    return avg_loss, cat_acc, cat_f1, pri_acc


def evaluate(model, dataloader, cat_criterion, pri_criterion, device):
    """
    Evaluate model on validation or test set.
    model.eval() disables dropout so we get consistent predictions.
    torch.no_grad() saves memory by not tracking gradients.
    """
    model.eval()
    total_loss = 0
    all_cat_preds, all_cat_labels = [], []
    all_pri_preds, all_pri_labels = [], []
    
    with torch.no_grad():  # No gradient computation during evaluation
        for texts, cat_labels, pri_labels in dataloader:
            texts = texts.to(device)
            cat_labels = cat_labels.to(device)
            pri_labels = pri_labels.to(device)
            
            cat_logits, pri_logits = model(texts)
            
            cat_loss = cat_criterion(cat_logits, cat_labels)
            pri_loss = pri_criterion(pri_logits, pri_labels)
            loss = cat_loss + pri_loss
            
            total_loss += loss.item()
            
            all_cat_preds.extend(cat_logits.argmax(dim=1).cpu().numpy())
            all_cat_labels.extend(cat_labels.cpu().numpy())
            all_pri_preds.extend(pri_logits.argmax(dim=1).cpu().numpy())
            all_pri_labels.extend(pri_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    cat_acc = accuracy_score(all_cat_labels, all_cat_preds)
    cat_f1 = f1_score(all_cat_labels, all_cat_preds, average="weighted")
    pri_acc = accuracy_score(all_pri_labels, all_pri_preds)
    pri_f1 = f1_score(all_pri_labels, all_pri_preds, average="weighted")
    
    return avg_loss, cat_acc, cat_f1, pri_acc, pri_f1, all_cat_preds, all_cat_labels, all_pri_preds, all_pri_labels


def compute_class_weights(labels, num_classes, device):
    """
    Compute weights for each class to handle imbalance.
    Classes with fewer samples get higher weights.
    Same concept as class_weight='balanced' in sklearn.
    """
    counts = np.bincount(labels, minlength=num_classes)
    # Avoid division by zero
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # Normalize
    return torch.FloatTensor(weights).to(device)


# ============================================================
# PART 4: Main Training Pipeline
# ============================================================

if __name__ == "__main__":
    print("SmartTicket - BiLSTM Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
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
    # Create Datasets and DataLoaders
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Creating datasets...")
    print("=" * 60)
    
    max_length = config["data"]["max_seq_length"]  # 64
    batch_size = config["models"]["bilstm"]["batch_size"]  # 32
    
    # Training dataset builds the vocabulary
    train_dataset = TicketDataset(
        train_df["text"], train_df["category_id"], train_df["priority_id"],
        vocab=None, max_length=max_length,
    )
    
    # Val and test datasets REUSE training vocabulary (no data leakage!)
    val_dataset = TicketDataset(
        val_df["text"], val_df["category_id"], val_df["priority_id"],
        vocab=train_dataset.vocab, max_length=max_length,
    )
    test_dataset = TicketDataset(
        test_df["text"], test_df["category_id"], test_df["priority_id"],
        vocab=train_dataset.vocab, max_length=max_length,
    )
    
    # DataLoaders: handle batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Vocabulary size: {len(train_dataset.vocab)}")
    print(f"  Max sequence length: {max_length}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    
    # Save vocabulary for later use in API
    with open("models/bilstm_vocab.json", "w") as f:
        json.dump(train_dataset.vocab, f)
    print(f"  Saved: models/bilstm_vocab.json")
    
    # ----------------------------------------------------------
    # Build Model
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building BiLSTM model...")
    print("=" * 60)
    
    model = BiLSTMClassifier(
        vocab_size=len(train_dataset.vocab),
        embedding_dim=config["models"]["bilstm"]["embedding_dim"],
        hidden_dim=config["models"]["bilstm"]["hidden_dim"],
        num_layers=config["models"]["bilstm"]["num_layers"],
        num_categories=10,
        num_priorities=4,
        dropout=config["models"]["bilstm"]["dropout"],
        fc_dropout=config["models"]["bilstm"]["fc_dropout"],
    ).to(DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model Architecture:")
    print(model)
    print(f"\n  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # ----------------------------------------------------------
    # Loss Functions, Optimizer, Scheduler
    # ----------------------------------------------------------
    
    # Class weights for imbalanced data
    cat_weights = compute_class_weights(
        train_df["category_id"].values, 10, DEVICE
    )
    pri_weights = compute_class_weights(
        train_df["priority_id"].values, 4, DEVICE
    )
    
    cat_criterion = nn.CrossEntropyLoss(weight=cat_weights)
    pri_criterion = nn.CrossEntropyLoss(weight=pri_weights)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["models"]["bilstm"]["learning_rate"],
    )
    
    # Reduce learning rate when validation loss stops improving
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )
    
    # ----------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    epochs = config["models"]["bilstm"]["epochs"]
    patience = config["models"]["bilstm"]["patience"]
    best_val_f1 = 0
    patience_counter = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_cat_acc": [], "val_cat_acc": [],
        "train_cat_f1": [], "val_cat_f1": [],
    }
    
    # MLflow tracking
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name="bilstm_multitask"):
        
        # Log parameters
        mlflow.log_param("model_type", "BiLSTM")
        mlflow.log_param("embedding_dim", config["models"]["bilstm"]["embedding_dim"])
        mlflow.log_param("hidden_dim", config["models"]["bilstm"]["hidden_dim"])
        mlflow.log_param("num_layers", config["models"]["bilstm"]["num_layers"])
        mlflow.log_param("dropout", config["models"]["bilstm"]["dropout"])
        mlflow.log_param("learning_rate", config["models"]["bilstm"]["learning_rate"])
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("vocab_size", len(train_dataset.vocab))
        
        training_start = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_cat_acc, train_cat_f1, train_pri_acc = train_one_epoch(
                model, train_loader, cat_criterion, pri_criterion, optimizer, DEVICE
            )
            
            # Validate
            val_loss, val_cat_acc, val_cat_f1, val_pri_acc, val_pri_f1, _, _, _, _ = evaluate(
                model, val_loader, cat_criterion, pri_criterion, DEVICE
            )
            
            epoch_time = time.time() - epoch_start
            
            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_cat_acc"].append(train_cat_acc)
            history["val_cat_acc"].append(val_cat_acc)
            history["train_cat_f1"].append(train_cat_f1)
            history["val_cat_f1"].append(val_cat_f1)
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_cat_f1", val_cat_f1, step=epoch)
            
            # Print progress
            print(f"\n  Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
            print(f"    Train Loss: {train_loss:.4f} | Cat Acc: {train_cat_acc:.4f} | Cat F1: {train_cat_f1:.4f}")
            print(f"    Val   Loss: {val_loss:.4f} | Cat Acc: {val_cat_acc:.4f} | Cat F1: {val_cat_f1:.4f} | Pri Acc: {val_pri_acc:.4f}")
            
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_cat_f1 > best_val_f1:
                best_val_f1 = val_cat_f1
                patience_counter = 0
                # Save best model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(train_dataset.vocab),
                    "config": config["models"]["bilstm"],
                }, "models/bilstm_best.pth")
                print(f"    ✅ New best model saved! (Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"    ⏳ No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\n  ⛔ Early stopping at epoch {epoch}")
                    break
        
        total_train_time = time.time() - training_start
        print(f"\n  Total training time: {total_train_time:.1f} seconds")
        
        # ----------------------------------------------------------
        # Final Evaluation on Test Set
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("Final Evaluation on Test Set")
        print("=" * 60)
        
        # Load best model
        checkpoint = torch.load("models/bilstm_best.pth", map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        test_loss, test_cat_acc, test_cat_f1, test_pri_acc, test_pri_f1, \
            cat_preds, cat_labels, pri_preds, pri_labels = evaluate(
                model, test_loader, cat_criterion, pri_criterion, DEVICE
            )
        
        # Category metrics
        id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
        category_names = [id_to_cat[i] for i in range(10)]
        
        print(f"\n  Category Classification:")
        print(f"    Test Accuracy: {test_cat_acc:.4f}")
        print(f"    Test F1 Score: {test_cat_f1:.4f}")
        print(f"\n{classification_report(cat_labels, cat_preds, target_names=category_names)}")
        
        # Priority metrics
        id_to_pri = {int(k): v for k, v in mappings["id_to_priority"].items()}
        priority_names = [id_to_pri[i] for i in range(4)]
        
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=category_names, yticklabels=category_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"BiLSTM Category Confusion Matrix (Test Accuracy: {test_cat_acc:.1%})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("reports/figures/bilstm_category_confusion_matrix.png", dpi=150)
        plt.close()
        mlflow.log_artifact("reports/figures/bilstm_category_confusion_matrix.png")
        
        # ----------------------------------------------------------
        # Training Curves
        # ----------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(history["val_loss"], label="Val Loss", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("BiLSTM Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history["train_cat_f1"], label="Train F1", marker="o")
        axes[1].plot(history["val_cat_f1"], label="Val F1", marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("BiLSTM Training & Validation F1")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig("reports/figures/bilstm_training_curves.png", dpi=150)
        plt.close()
        mlflow.log_artifact("reports/figures/bilstm_training_curves.png")
        
        # ----------------------------------------------------------
        # Inference Speed Benchmark
        # ----------------------------------------------------------
        model.eval()
        sample_input = torch.randint(0, 100, (1, max_length)).to(DEVICE)
        
        # Warm up GPU
        for _ in range(10):
            with torch.no_grad():
                model(sample_input)
        
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                model(sample_input)
        avg_ms = (time.time() - start) / 100 * 1000
        
        mlflow.log_metric("inference_ms", avg_ms)
        
        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------
        print("\n" + "=" * 60)
        print("BiLSTM TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\n  Category Model:")
        print(f"    Test Accuracy: {test_cat_acc:.1%}")
        print(f"    Test F1 Score: {test_cat_f1:.4f}")
        print(f"\n  Priority Model:")
        print(f"    Test Accuracy: {test_pri_acc:.1%}")
        print(f"    Test F1 Score: {test_pri_f1:.4f}")
        print(f"\n  Inference: {avg_ms:.2f} ms/query")
        print(f"  Training time: {total_train_time:.1f}s")
        print(f"  Best model saved: models/bilstm_best.pth")
        print(f"\n  Comparison with baseline:")
        print(f"    Baseline:  87.5% accuracy, 0.8752 F1")
        print(f"    BiLSTM:    {test_cat_acc:.1%} accuracy, {test_cat_f1:.4f} F1")