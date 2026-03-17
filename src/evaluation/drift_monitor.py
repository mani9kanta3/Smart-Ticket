"""
SmartTicket - Data Drift Monitoring with Evidently AI
Compares training data distribution vs incoming data to detect
when the model might need retraining.

What is drift?
- Training data: customers said "transfer", "payment", "card"
- New data: customers now say "UPI", "GPay", "contactless"
- The model hasn't seen these new words → accuracy drops
- Drift detection catches this BEFORE accuracy drops significantly
"""

import os
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    ColumnDistributionMetric,
)

os.makedirs("reports/drift_reports", exist_ok=True)


def prepare_reference_data():
    """
    Load training data as our 'reference' — this is what the model
    was trained on and what it expects to see.
    """
    train_df = pd.read_csv("data/processed/train.csv")
    
    # Add text length feature (useful for drift detection)
    train_df["text_length"] = train_df["text"].str.len()
    train_df["word_count"] = train_df["text"].str.split().str.len()
    
    return train_df


def prepare_current_data():
    """
    Load test data as simulated 'production' data.
    In a real system, this would be live incoming tickets.
    
    We also create a 'drifted' version to show what drift looks like.
    """
    test_df = pd.read_csv("data/processed/test.csv")
    test_df["text_length"] = test_df["text"].str.len()
    test_df["word_count"] = test_df["text"].str.split().str.len()
    
    return test_df


def create_drifted_data():
    """
    Simulate drifted data — what happens when customer language changes.
    This demonstrates why monitoring matters.
    """
    test_df = pd.read_csv("data/processed/test.csv")
    
    drifted_texts = []
    for text in test_df["text"]:
        # Simulate language drift by replacing common words
        modified = text.lower()
        modified = modified.replace("transfer", "UPI payment")
        modified = modified.replace("card", "virtual wallet")
        modified = modified.replace("payment", "transaction via app")
        modified = modified.replace("bank", "fintech platform")
        # Add extra words to simulate longer queries (trend shift)
        modified = modified + " please help me with this issue as soon as possible"
        drifted_texts.append(modified)
    
    drifted_df = test_df.copy()
    drifted_df["text"] = drifted_texts
    drifted_df["text_length"] = drifted_df["text"].str.len()
    drifted_df["word_count"] = drifted_df["text"].str.split().str.len()
    
    return drifted_df


def generate_drift_report(reference_df, current_df, report_name="drift_report"):
    """
    Generate an Evidently AI drift report comparing
    reference (training) data vs current (production) data.
    """
    print(f"\n  Generating drift report: {report_name}...")
    print(f"    Reference data: {len(reference_df)} samples")
    print(f"    Current data:   {len(current_df)} samples")
    
    # Create report with drift metrics
    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name="text_length"),
        ColumnDriftMetric(column_name="word_count"),
        ColumnDriftMetric(column_name="category_id"),
        ColumnDistributionMetric(column_name="text_length"),
        ColumnDistributionMetric(column_name="word_count"),
    ])
    
    report.run(
        reference_data=reference_df[["text_length", "word_count", "category_id"]],
        current_data=current_df[["text_length", "word_count", "category_id"]],
    )
    
    # Save HTML report
    report_path = f"reports/drift_reports/{report_name}.html"
    report.save_html(report_path)
    print(f"    Saved: {report_path}")
    
    # Extract drift results
    report_dict = report.as_dict()
    
    return report_path, report_dict


def analyze_drift_results(report_dict, report_name):
    """Analyze and print drift detection results."""
    print(f"\n  Drift Analysis for: {report_name}")
    print(f"  {'-'*50}")
    
    metrics = report_dict.get("metrics", [])
    
    for metric in metrics:
        metric_id = metric.get("metric", "")
        result = metric.get("result", {})
        
        if "DatasetDriftMetric" in metric_id:
            is_drift = result.get("dataset_drift", False)
            drift_share = result.get("drift_share", 0)
            print(f"    Dataset Drift Detected: {'YES ⚠️' if is_drift else 'NO ✅'}")
            print(f"    Drift Share: {drift_share:.1%}")
        
        elif "ColumnDriftMetric" in metric_id:
            col = result.get("column_name", "unknown")
            is_drift = result.get("drift_detected", False)
            drift_score = result.get("drift_score", 0)
            status = "⚠️ DRIFT" if is_drift else "✅ OK"
            print(f"    {col}: drift_score={drift_score:.4f} {status}")


if __name__ == "__main__":
    print("SmartTicket - Data Drift Monitoring")
    print("=" * 60)
    
    # Load reference data (training set)
    print("\nLoading reference data (training set)...")
    reference_df = prepare_reference_data()
    
    # ----------------------------------------------------------
    # Report 1: Normal conditions (test data ≈ training data)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("REPORT 1: Normal Conditions (no drift expected)")
    print("=" * 60)
    
    current_df = prepare_current_data()
    
    report_path_1, report_dict_1 = generate_drift_report(
        reference_df, current_df, "normal_conditions"
    )
    analyze_drift_results(report_dict_1, "Normal Conditions")
    
    # ----------------------------------------------------------
    # Report 2: Drifted data (simulated language change)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("REPORT 2: Drifted Data (simulated language change)")
    print("=" * 60)
    
    drifted_df = create_drifted_data()
    
    report_path_2, report_dict_2 = generate_drift_report(
        reference_df, drifted_df, "drifted_conditions"
    )
    analyze_drift_results(report_dict_2, "Drifted Conditions")
    
    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("DRIFT MONITORING COMPLETE!")
    print("=" * 60)
    print(f"\n  Reports saved:")
    print(f"    1. reports/drift_reports/normal_conditions.html")
    print(f"    2. reports/drift_reports/drifted_conditions.html")
    print(f"\n  Open these HTML files in your browser to see detailed visualizations.")
    print(f"\n  In production, you would:")
    print(f"    - Run this daily/weekly on incoming ticket data")
    print(f"    - Alert when drift_score > 0.15")
    print(f"    - Trigger model retraining when significant drift detected")