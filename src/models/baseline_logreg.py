"""
SmartTicket - Baseline Model: TF-IDF + Logistic Regression
This is our accuracy FLOOR. Deep learning models must beat this.

How it works:
1. TF-IDF converts text → numbers (each query becomes a 10,000-dim vector)
2. Logistic Regression draws decision boundaries to separate categories
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import mlflow
import mlflow.sklearn
import joblib
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_data():
    """Load the prepared train/val/test splits."""
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df, mappings


def build_tfidf(train_texts, val_texts, test_texts):
    """
    Convert text to TF-IDF vectors.
    
    TF-IDF = Term Frequency × Inverse Document Frequency
    - Words that appear often in ONE query but rarely across ALL queries get HIGH scores
    - Common words like "the", "is" get LOW scores (appear everywhere = not useful)
    - ngram_range=(1,2) captures both single words AND two-word phrases
      e.g., "not working" as a bigram is more informative than "not" and "working" separately
    """
    print("\n" + "=" * 60)
    print("STEP 1: TF-IDF Vectorization")
    print("=" * 60)
    
    tfidf = TfidfVectorizer(
        max_features=config["models"]["baseline"]["max_features"],  # Top 10,000 words
        ngram_range=tuple(config["models"]["baseline"]["ngram_range"]),  # Unigrams + Bigrams
        sublinear_tf=True,  # Apply log to term frequency (reduces impact of very frequent words)
        strip_accents="unicode",
        lowercase=True,
    )
    
    # Fit on training data ONLY (no data leakage!)
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)
    
    print(f"\n  TF-IDF matrix shapes:")
    print(f"    Train: {X_train.shape} (samples × features)")
    print(f"    Val:   {X_val.shape}")
    print(f"    Test:  {X_test.shape}")
    print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")
    
    # Show top features (most informative words)
    feature_names = tfidf.get_feature_names_out()
    print(f"\n  Sample features: {list(feature_names[:10])}")
    print(f"  Sample bigrams: {[f for f in feature_names if ' ' in f][:10]}")
    
    # Save vectorizer for later use in API
    joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
    print(f"\n  Saved: models/tfidf_vectorizer.joblib")
    
    return tfidf, X_train, X_val, X_test


def train_category_model(X_train, y_train, X_val, y_val, X_test, y_test, mappings):
    """
    Train Logistic Regression for category classification (10 classes).
    
    class_weight='balanced': gives more importance to underrepresented categories.
    Without it, model would be biased toward 'General Inquiry' (19.5% of data).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Category Classification (Logistic Regression)")
    print("=" * 60)
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,  # Regularization strength (1.0 = default)
        solver="lbfgs",
        random_state=config["data"]["random_state"],
    )
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"\n  Training time: {train_time:.2f} seconds")
    
    # Predict on all sets
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Metrics
    id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
    category_names = [id_to_cat[i] for i in sorted(id_to_cat.keys())]
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    test_f1 = f1_score(y_test, test_pred, average="weighted")
    test_precision = precision_score(y_test, test_pred, average="weighted")
    test_recall = recall_score(y_test, test_pred, average="weighted")
    
    print(f"\n  Accuracy:")
    print(f"    Train: {train_acc:.4f}")
    print(f"    Val:   {val_acc:.4f}")
    print(f"    Test:  {test_acc:.4f}")
    print(f"\n  Test Metrics:")
    print(f"    Weighted F1:        {test_f1:.4f}")
    print(f"    Weighted Precision: {test_precision:.4f}")
    print(f"    Weighted Recall:    {test_recall:.4f}")
    
    # Per-class report
    print(f"\n  Per-Class Classification Report:")
    print(classification_report(y_test, test_pred, target_names=category_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=category_names,
        yticklabels=category_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Baseline Category Confusion Matrix (Test Accuracy: {test_acc:.1%})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/figures/baseline_category_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: reports/figures/baseline_category_confusion_matrix.png")
    
    # Inference speed benchmark
    start_time = time.time()
    for _ in range(100):
        model.predict(X_test[:1])
    avg_inference_ms = (time.time() - start_time) / 100 * 1000
    print(f"\n  Average inference time: {avg_inference_ms:.2f} ms per query")
    
    # Save model
    joblib.dump(model, "models/baseline_category_model.joblib")
    print(f"  Saved: models/baseline_category_model.joblib")
    
    metrics = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "inference_ms": avg_inference_ms,
        "train_time_sec": train_time,
    }
    
    return model, metrics


def train_priority_model(X_train, y_train, X_val, y_val, X_test, y_test, mappings):
    """
    Train Logistic Regression for priority classification (4 classes: P0-P3).
    Same approach as category, but predicting urgency instead.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Priority Classification (Logistic Regression)")
    print("=" * 60)
    
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=config["data"]["random_state"],
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    test_pred = model.predict(X_test)
    
    id_to_pri = {int(k): v for k, v in mappings["id_to_priority"].items()}
    priority_names = [id_to_pri[i] for i in sorted(id_to_pri.keys())]
    
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="weighted")
    
    print(f"\n  Test Accuracy:    {test_acc:.4f}")
    print(f"  Test Weighted F1: {test_f1:.4f}")
    print(f"\n  Per-Class Classification Report:")
    print(classification_report(y_test, test_pred, target_names=priority_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=priority_names,
        yticklabels=priority_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Baseline Priority Confusion Matrix (Test Accuracy: {test_acc:.1%})")
    plt.tight_layout()
    plt.savefig("reports/figures/baseline_priority_confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved: reports/figures/baseline_priority_confusion_matrix.png")
    
    # Save model
    joblib.dump(model, "models/baseline_priority_model.joblib")
    print(f"  Saved: models/baseline_priority_model.joblib")
    
    metrics = {
        "test_acc": test_acc,
        "test_f1": test_f1,
        "train_time_sec": train_time,
    }
    
    return model, metrics


def show_misclassifications(model, tfidf, test_df, y_test, mappings, n=10):
    """
    Show examples where the model got it wrong.
    This helps us understand WHERE the model struggles.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Error Analysis")
    print("=" * 60)
    
    X_test = tfidf.transform(test_df["text"])
    predictions = model.predict(X_test)
    
    id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
    
    # Find misclassified examples
    wrong_mask = predictions != y_test
    wrong_indices = np.where(wrong_mask)[0]
    
    print(f"\n  Total misclassified: {len(wrong_indices)} / {len(y_test)} ({len(wrong_indices)/len(y_test)*100:.1f}%)")
    print(f"\n  Sample misclassifications:")
    
    for idx in wrong_indices[:n]:
        text = test_df.iloc[idx]["text"]
        true_label = id_to_cat[y_test.iloc[idx]]
        pred_label = id_to_cat[predictions[idx]]
        print(f"\n    Text: \"{text}\"")
        print(f"    True: {true_label}  |  Predicted: {pred_label}")


if __name__ == "__main__":
    print("SmartTicket - Baseline Model Training")
    print("=" * 60)
    
    # Load data
    train_df, val_df, test_df, mappings = load_data()
    
    # Build TF-IDF features
    tfidf, X_train, X_val, X_test = build_tfidf(
        train_df["text"], val_df["text"], test_df["text"]
    )
    
    # ============================================================
    # MLflow Experiment Tracking
    # ============================================================
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run(run_name="baseline_tfidf_logreg"):
        
        # Log parameters
        mlflow.log_param("model_type", "TF-IDF + Logistic Regression")
        mlflow.log_param("max_features", config["models"]["baseline"]["max_features"])
        mlflow.log_param("ngram_range", str(config["models"]["baseline"]["ngram_range"]))
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("num_categories", 10)
        
        # Train category model
        cat_model, cat_metrics = train_category_model(
            X_train, train_df["category_id"],
            X_val, val_df["category_id"],
            X_test, test_df["category_id"],
            mappings,
        )
        
        # Train priority model
        pri_model, pri_metrics = train_priority_model(
            X_train, train_df["priority_id"],
            X_val, val_df["priority_id"],
            X_test, test_df["priority_id"],
            mappings,
        )
        
        # Log all metrics to MLflow
        mlflow.log_metric("category_test_accuracy", cat_metrics["test_acc"])
        mlflow.log_metric("category_test_f1", cat_metrics["test_f1"])
        mlflow.log_metric("category_test_precision", cat_metrics["test_precision"])
        mlflow.log_metric("category_test_recall", cat_metrics["test_recall"])
        mlflow.log_metric("category_inference_ms", cat_metrics["inference_ms"])
        mlflow.log_metric("priority_test_accuracy", pri_metrics["test_acc"])
        mlflow.log_metric("priority_test_f1", pri_metrics["test_f1"])
        
        # Log models
        mlflow.sklearn.log_model(cat_model, "category_model")
        mlflow.sklearn.log_model(pri_model, "priority_model")
        
        # Log figures
        mlflow.log_artifact("reports/figures/baseline_category_confusion_matrix.png")
        mlflow.log_artifact("reports/figures/baseline_priority_confusion_matrix.png")
        
        # Error analysis
        show_misclassifications(cat_model, tfidf, test_df, test_df["category_id"], mappings)
        
        print("\n" + "=" * 60)
        print("BASELINE TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\n  Category Model:")
        print(f"    Test Accuracy: {cat_metrics['test_acc']:.1%}")
        print(f"    Test F1 Score: {cat_metrics['test_f1']:.4f}")
        print(f"    Inference:     {cat_metrics['inference_ms']:.2f} ms/query")
        print(f"\n  Priority Model:")
        print(f"    Test Accuracy: {pri_metrics['test_acc']:.1%}")
        print(f"    Test F1 Score: {pri_metrics['test_f1']:.4f}")
        print(f"\n  MLflow run logged successfully!")
        print(f"  Run 'mlflow ui' to view experiments in browser")