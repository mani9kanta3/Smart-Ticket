"""
SmartTicket - Unit Tests
Tests for API schemas and basic functionality.
"""

import pytest
import json
import os


def test_label_mappings_exist():
    """Check that label mappings file exists and is valid."""
    path = "data/processed/label_mappings.json"
    assert os.path.exists(path), "label_mappings.json not found"
    
    with open(path, "r") as f:
        mappings = json.load(f)
    
    assert "id_to_category" in mappings
    assert "id_to_priority" in mappings
    assert len(mappings["id_to_category"]) == 10
    assert len(mappings["id_to_priority"]) == 4


def test_category_names():
    """Verify all 10 categories are present."""
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    expected_categories = {
        "Account Access", "Account Management", "Balance & Statement",
        "Card Services", "Fees & Charges", "General Inquiry",
        "Payment Issues", "Refund & Dispute", "Technical Issues",
        "Transfer & Transaction",
    }
    
    actual_categories = set(mappings["id_to_category"].values())
    assert actual_categories == expected_categories


def test_priority_names():
    """Verify all 4 priorities are present."""
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    expected = {"P0-Critical", "P1-High", "P2-Medium", "P3-Low"}
    actual = set(mappings["id_to_priority"].values())
    assert actual == expected


def test_config_file():
    """Check config file loads correctly."""
    import yaml
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    assert config["project"]["name"] == "SmartTicket"
    assert config["data"]["num_categories"] == 10
    assert config["models"]["bert"]["model_name"] == "distilbert-base-uncased"


def test_model_comparison_exists():
    """Check model comparison data exists."""
    path = "models/model_comparison.json"
    assert os.path.exists(path), "model_comparison.json not found"
    
    with open(path, "r") as f:
        data = json.load(f)
    
    assert len(data) == 4, "Should have 4 models in comparison"


def test_onnx_model_exists():
    """Check ONNX model file exists."""
    assert os.path.exists("models/smartticket_bert.onnx"), "ONNX model not found"


def test_tfidf_model_exists():
    """Check baseline model files exist."""
    assert os.path.exists("models/tfidf_vectorizer.joblib")
    assert os.path.exists("models/baseline_category_model.joblib")
    assert os.path.exists("models/baseline_priority_model.joblib")


def test_train_data_exists():
    """Check processed data files exist."""
    assert os.path.exists("data/processed/train.csv")
    assert os.path.exists("data/processed/val.csv")
    assert os.path.exists("data/processed/test.csv")


def test_train_data_shape():
    """Verify training data has expected columns."""
    import pandas as pd
    
    train_df = pd.read_csv("data/processed/train.csv")
    
    required_columns = ["text", "category", "category_id", "priority", "priority_id"]
    for col in required_columns:
        assert col in train_df.columns, f"Missing column: {col}"
    
    assert len(train_df) > 10000, "Training set seems too small"
    assert train_df["category_id"].nunique() == 10
    assert train_df["priority_id"].nunique() == 4