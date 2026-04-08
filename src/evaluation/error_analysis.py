"""
SmartTicket - Class-wise Error Analysis
For each category, analyzes:
  - Per-class precision, recall, F1
  - Top confused pairs (which categories get mixed up)
  - Sample misclassified queries with explanations
  - Confidence distribution for correct vs wrong predictions

This is what interviewers want to see — not just "90.6% accuracy"
but WHERE the model fails and WHY.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_data():
    """Load test data and the baseline model."""
    test_df = pd.read_csv("data/processed/test.csv")
    
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)
    
    tfidf = joblib.load("models/tfidf_vectorizer.joblib")
    cat_model = joblib.load("models/baseline_category_model.joblib")
    
    return test_df, mappings, tfidf, cat_model


def get_predictions(test_df, tfidf, cat_model):
    """Get predictions and confidence scores for the entire test set."""
    X = tfidf.transform(test_df["text"])
    predictions = cat_model.predict(X)
    probabilities = cat_model.predict_proba(X)
    confidences = probabilities.max(axis=1)
    
    return predictions, probabilities, confidences


def per_class_metrics(test_df, predictions, mappings):
    """Calculate per-class precision, recall, F1."""
    print("=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    
    id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
    category_names = [id_to_cat[i] for i in range(10)]
    
    report = classification_report(
        test_df["category_id"],
        predictions,
        target_names=category_names,
        output_dict=True,
    )
    
    # Sort by F1 score
    class_metrics = []
    for cat in category_names:
        if cat in report:
            class_metrics.append({
                "category": cat,
                "precision": report[cat]["precision"],
                "recall": report[cat]["recall"],
                "f1": report[cat]["f1-score"],
                "support": int(report[cat]["support"]),
            })
    
    df = pd.DataFrame(class_metrics).sort_values("f1", ascending=True)
    
    print(f"\n  {'Category':<25s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'-'*70}")
    for _, row in df.iterrows():
        print(f"  {row['category']:<25s} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f} {row['support']:>10d}")
    
    print(f"\n  WORST PERFORMING:  {df.iloc[0]['category']} (F1={df.iloc[0]['f1']:.4f})")
    print(f"  BEST PERFORMING:   {df.iloc[-1]['category']} (F1={df.iloc[-1]['f1']:.4f})")
    
    return df


def confusion_pairs(test_df, predictions, mappings, top_n=10):
    """Find the most common confusion pairs (true → predicted)."""
    print("\n" + "=" * 70)
    print("TOP CONFUSION PAIRS (where the model gets confused most)")
    print("=" * 70)
    
    id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
    
    confusions = []
    for true_id, pred_id in zip(test_df["category_id"], predictions):
        if true_id != pred_id:
            confusions.append((id_to_cat[true_id], id_to_cat[pred_id]))
    
    confusion_counts = Counter(confusions).most_common(top_n)
    
    print(f"\n  {'True Category':<25s} → {'Predicted Category':<25s} {'Count':>6s}")
    print(f"  {'-'*65}")
    for (true_cat, pred_cat), count in confusion_counts:
        print(f"  {true_cat:<25s} → {pred_cat:<25s} {count:>6d}")
    
    return confusion_counts


def confidence_analysis(test_df, predictions, confidences, mappings):
    """Analyze confidence distribution for correct vs incorrect predictions."""
    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)
    
    correct = predictions == test_df["category_id"].values
    
    correct_confidences = confidences[correct]
    wrong_confidences = confidences[~correct]
    
    print(f"\n  Correct predictions: {len(correct_confidences)} ({correct.mean()*100:.1f}%)")
    print(f"    Mean confidence:   {correct_confidences.mean():.4f}")
    print(f"    Median confidence: {np.median(correct_confidences):.4f}")
    
    print(f"\n  Wrong predictions:   {len(wrong_confidences)} ({(~correct).mean()*100:.1f}%)")
    print(f"    Mean confidence:   {wrong_confidences.mean():.4f}")
    print(f"    Median confidence: {np.median(wrong_confidences):.4f}")
    
    # Threshold analysis
    print(f"\n  THRESHOLD ANALYSIS (what % of wrong predictions could be caught?)")
    print(f"  {'Threshold':<12s} {'Auto-routed':>15s} {'Sent to Review':>17s} {'Accuracy on Auto':>20s}")
    print(f"  {'-'*70}")
    
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        above_threshold = confidences >= threshold
        auto_routed = above_threshold.sum()
        sent_to_review = (~above_threshold).sum()
        if auto_routed > 0:
            auto_accuracy = correct[above_threshold].mean()
        else:
            auto_accuracy = 0
        print(f"  {threshold:<12.2f} {auto_routed:>10d} ({auto_routed/len(test_df)*100:>4.1f}%)  {sent_to_review:>10d} ({sent_to_review/len(test_df)*100:>4.1f}%)  {auto_accuracy:>18.1%}")
    
    return correct_confidences, wrong_confidences


def sample_misclassifications(test_df, predictions, confidences, mappings, n=15):
    """Show examples of wrong predictions with explanations."""
    print("\n" + "=" * 70)
    print(f"SAMPLE MISCLASSIFICATIONS (top {n} highest-confidence mistakes)")
    print("=" * 70)
    print("\nThese are predictions the model was CONFIDENT about but got WRONG.")
    print("These are the most informative errors to study.\n")
    
    id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
    
    test_df = test_df.reset_index(drop=True)
    wrong_mask = predictions != test_df["category_id"].values
    wrong_indices = np.where(wrong_mask)[0]
    
    # Sort by confidence (highest confidence wrong predictions first)
    wrong_with_conf = [(i, confidences[i]) for i in wrong_indices]
    wrong_with_conf.sort(key=lambda x: -x[1])
    
    for rank, (idx, conf) in enumerate(wrong_with_conf[:n], 1):
        text = test_df.iloc[idx]["text"]
        true_cat = id_to_cat[test_df.iloc[idx]["category_id"]]
        pred_cat = id_to_cat[predictions[idx]]
        
        print(f"\n  [{rank}] Confidence: {conf:.1%}")
        print(f"      Text: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"      True:      {true_cat}")
        print(f"      Predicted: {pred_cat}")
        # Brief explanation hint
        if true_cat == "Payment Issues" and pred_cat == "Fees & Charges":
            print(f"      Why: Both involve money — model focused on charge keywords")
        elif true_cat == "Account Access" and pred_cat == "Card Services":
            print(f"      Why: Card-related security; model saw 'card' keyword")


def save_visualizations(test_df, predictions, confidences, mappings, class_df):
    """Save error analysis charts."""
    os.makedirs("reports/figures", exist_ok=True)
    id_to_cat = {int(k): v for k, v in mappings["id_to_category"].items()}
    category_names = [id_to_cat[i] for i in range(10)]
    
    # Per-class F1 chart
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = class_df.sort_values("f1")
    colors = ["#d32f2f" if f < 0.85 else "#f57c00" if f < 0.9 else "#388e3c" for f in df_sorted["f1"]]
    ax.barh(df_sorted["category"], df_sorted["f1"], color=colors)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score (Red < 0.85, Orange < 0.90, Green ≥ 0.90)")
    ax.set_xlim(0, 1)
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(row["f1"] + 0.01, i, f"{row['f1']:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("reports/figures/error_analysis_per_class_f1.png", dpi=150)
    plt.close()
    print("\n  Saved: reports/figures/error_analysis_per_class_f1.png")
    
    # Confidence distribution
    correct = predictions == test_df["category_id"].values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences[correct], bins=30, alpha=0.6, label="Correct", color="#388e3c")
    ax.hist(confidences[~correct], bins=30, alpha=0.6, label="Wrong", color="#d32f2f")
    ax.axvline(0.85, color="black", linestyle="--", label="0.85 threshold")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs Wrong Predictions")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/error_analysis_confidence_distribution.png", dpi=150)
    plt.close()
    print("  Saved: reports/figures/error_analysis_confidence_distribution.png")


if __name__ == "__main__":
    print("SmartTicket - Class-wise Error Analysis")
    print("=" * 70)
    print("\nAnalyzing where the baseline model fails and why.\n")
    
    test_df, mappings, tfidf, cat_model = load_data()
    predictions, probabilities, confidences = get_predictions(test_df, tfidf, cat_model)
    
    class_df = per_class_metrics(test_df, predictions, mappings)
    confusion_counts = confusion_pairs(test_df, predictions, mappings)
    correct_conf, wrong_conf = confidence_analysis(test_df, predictions, confidences, mappings)
    sample_misclassifications(test_df, predictions, confidences, mappings)
    save_visualizations(test_df, predictions, confidences, mappings, class_df)
    
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\n  KEY TAKEAWAYS FOR INTERVIEWS:")
    print(f"    1. Worst performing category: {class_df.iloc[0]['category']} (F1={class_df.iloc[0]['f1']:.3f})")
    print(f"    2. Top confusion: {confusion_counts[0][0][0]} → {confusion_counts[0][0][1]} ({confusion_counts[0][1]} times)")
    print(f"    3. Wrong predictions have lower confidence on average ({wrong_conf.mean():.3f} vs {correct_conf.mean():.3f})")
    print(f"    4. With 0.85 threshold, ~{(confidences >= 0.85).sum() / len(test_df) * 100:.0f}% of tickets can be auto-routed safely")