"""
SmartTicket - Exploratory Data Analysis (EDA)
Analyzes Banking77 and CLINC150 datasets before model training.
Generates charts saved to reports/figures/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Create output directory
os.makedirs("reports/figures", exist_ok=True)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11


def load_data():
    """Load the raw datasets."""
    banking = pd.read_csv("data/raw/banking77.csv")
    clinc = pd.read_csv("data/raw/clinc150.csv")
    print(f"Banking77 shape: {banking.shape}")
    print(f"CLINC150 shape: {clinc.shape}")
    return banking, clinc


def analyze_class_distribution(banking):
    """
    Analysis 1: How many samples per category?
    If some categories have way more samples than others,
    our model will be biased toward the bigger categories.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Class Distribution (Banking77)")
    print("=" * 60)
    
    # Count samples per category
    label_counts = banking["label_name"].value_counts()
    
    print(f"\nLargest category:  {label_counts.index[0]} ({label_counts.iloc[0]} samples)")
    print(f"Smallest category: {label_counts.index[-1]} ({label_counts.iloc[-1]} samples)")
    print(f"Imbalance ratio:   {label_counts.iloc[0] / label_counts.iloc[-1]:.1f}x")
    
    # Plot top 20 categories
    fig, ax = plt.subplots(figsize=(14, 8))
    label_counts.head(20).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of Samples")
    ax.set_title("Banking77: Top 20 Categories by Sample Count")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/figures/banking77_class_distribution.png", dpi=150)
    plt.close()
    print("Saved: reports/figures/banking77_class_distribution.png")
    
    # Plot bottom 20 categories
    fig, ax = plt.subplots(figsize=(14, 8))
    label_counts.tail(20).plot(kind="barh", ax=ax, color="coral")
    ax.set_xlabel("Number of Samples")
    ax.set_title("Banking77: Bottom 20 Categories by Sample Count")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("reports/figures/banking77_class_distribution_bottom.png", dpi=150)
    plt.close()
    print("Saved: reports/figures/banking77_class_distribution_bottom.png")
    
    return label_counts


def analyze_query_lengths(banking, clinc):
    """
    Analysis 2: How long are the queries?
    This helps us decide max_length for our models.
    Short queries = simpler models work fine.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Query Length Distribution")
    print("=" * 60)
    
    banking["word_count"] = banking["text"].str.split().str.len()
    clinc["word_count"] = clinc["text"].str.split().str.len()
    
    print(f"\nBanking77 query lengths:")
    print(f"  Mean:   {banking['word_count'].mean():.1f} words")
    print(f"  Median: {banking['word_count'].median():.1f} words")
    print(f"  Min:    {banking['word_count'].min()} words")
    print(f"  Max:    {banking['word_count'].max()} words")
    print(f"  95th percentile: {banking['word_count'].quantile(0.95):.0f} words")
    
    print(f"\nCLINC150 query lengths:")
    print(f"  Mean:   {clinc['word_count'].mean():.1f} words")
    print(f"  Median: {clinc['word_count'].median():.1f} words")
    print(f"  Min:    {clinc['word_count'].min()} words")
    print(f"  Max:    {clinc['word_count'].max()} words")
    
    # Plot both distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(banking["word_count"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Number of Words")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Banking77: Query Length Distribution")
    axes[0].axvline(banking["word_count"].mean(), color="red", linestyle="--", label=f"Mean: {banking['word_count'].mean():.1f}")
    axes[0].legend()
    
    axes[1].hist(clinc["word_count"], bins=30, color="coral", edgecolor="white")
    axes[1].set_xlabel("Number of Words")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("CLINC150: Query Length Distribution")
    axes[1].axvline(clinc["word_count"].mean(), color="red", linestyle="--", label=f"Mean: {clinc['word_count'].mean():.1f}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("reports/figures/query_length_distribution.png", dpi=150)
    plt.close()
    print("\nSaved: reports/figures/query_length_distribution.png")


def analyze_data_quality(banking, clinc):
    """
    Analysis 3: Check for data quality issues.
    Duplicates and empty strings can mess up training.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Data Quality Check")
    print("=" * 60)
    
    # Check duplicates
    banking_dupes = banking["text"].duplicated().sum()
    clinc_dupes = clinc["text"].duplicated().sum()
    
    print(f"\nBanking77 duplicates: {banking_dupes} ({banking_dupes/len(banking)*100:.1f}%)")
    print(f"CLINC150 duplicates:  {clinc_dupes} ({clinc_dupes/len(clinc)*100:.1f}%)")
    
    # Check empty strings
    banking_empty = (banking["text"].str.strip() == "").sum()
    clinc_empty = (clinc["text"].str.strip() == "").sum()
    
    print(f"\nBanking77 empty strings: {banking_empty}")
    print(f"CLINC150 empty strings:  {clinc_empty}")
    
    # Check for null values
    print(f"\nBanking77 null values: {banking['text'].isnull().sum()}")
    print(f"CLINC150 null values:  {clinc['text'].isnull().sum()}")


def analyze_vocabulary(banking):
    """
    Analysis 4: What words appear most frequently?
    Helps us understand what kind of language customers use.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Vocabulary Analysis")
    print("=" * 60)
    
    # Build word frequency
    all_words = " ".join(banking["text"].str.lower()).split()
    word_counts = Counter(all_words)
    
    print(f"\nTotal words: {len(all_words)}")
    print(f"Unique words: {len(word_counts)}")
    print(f"\nTop 20 most common words:")
    for word, count in word_counts.most_common(20):
        print(f"  {word:20s} {count:5d}")
    
    # Plot top 30 words
    top_words = word_counts.most_common(30)
    words, counts = zip(*top_words)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(words)), counts, color="steelblue")
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_ylabel("Frequency")
    ax.set_title("Banking77: Top 30 Most Common Words")
    plt.tight_layout()
    plt.savefig("reports/figures/vocabulary_top_words.png", dpi=150)
    plt.close()
    print("\nSaved: reports/figures/vocabulary_top_words.png")


def analyze_samples_per_category(banking):
    """
    Analysis 5: Show sample queries from different categories.
    This helps us understand what each category looks like.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Sample Queries by Category")
    print("=" * 60)
    
    # Show 2 samples from 10 different categories
    categories = banking["label_name"].unique()[:10]
    for cat in categories:
        samples = banking[banking["label_name"] == cat]["text"].head(2).tolist()
        print(f"\n  [{cat}]")
        for s in samples:
            print(f"    → {s}")


def analyze_clinc_oos(clinc):
    """
    Analysis 6: Analyze out-of-scope queries from CLINC150.
    These become our 'General Inquiry' category.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 6: CLINC150 Out-of-Scope Analysis")
    print("=" * 60)
    
    oos = clinc[clinc["intent_name"] == "oos"]
    in_scope = clinc[clinc["intent_name"] != "oos"]
    
    print(f"\nOut-of-scope: {len(oos)} queries")
    print(f"In-scope:     {len(in_scope)} queries")
    print(f"OOS ratio:    {len(oos)/len(clinc)*100:.1f}%")
    
    # Plot in-scope vs out-of-scope
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["In-Scope", "Out-of-Scope"], [len(in_scope), len(oos)], 
           color=["steelblue", "coral"])
    ax.set_ylabel("Number of Queries")
    ax.set_title("CLINC150: In-Scope vs Out-of-Scope Distribution")
    for i, v in enumerate([len(in_scope), len(oos)]):
        ax.text(i, v + 200, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/clinc150_oos_distribution.png", dpi=150)
    plt.close()
    print("Saved: reports/figures/clinc150_oos_distribution.png")
    
    print(f"\nSample OOS queries:")
    for text in oos["text"].sample(10, random_state=42).tolist():
        print(f"  → {text}")


if __name__ == "__main__":
    print("SmartTicket - Exploratory Data Analysis")
    print("=" * 60)
    
    # Load data
    banking, clinc = load_data()
    
    # Run all analyses
    label_counts = analyze_class_distribution(banking)
    analyze_query_lengths(banking, clinc)
    analyze_data_quality(banking, clinc)
    analyze_vocabulary(banking)
    analyze_samples_per_category(banking)
    analyze_clinc_oos(clinc)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE! Check reports/figures/ for all charts.")
    print("=" * 60)