"""
SmartTicket - Dataset Download Script
Downloads Banking77 and CLINC150 datasets from HuggingFace
and saves them as CSV files in data/raw/
"""

import os
import pandas as pd
from datasets import load_dataset

def download_banking77():
    """
    Download Banking77 dataset.
    Contains 13,083 customer banking queries across 77 intent categories.
    """
    print("=" * 60)
    print("Downloading Banking77 dataset...")
    print("=" * 60)
    
    # Load from HuggingFace
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    
    # Extract train and test splits
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Convert to DataFrames
    train_df = pd.DataFrame({
        "text": train_data["text"],
        "label": train_data["label"]
    })
    
    test_df = pd.DataFrame({
        "text": test_data["text"],
        "label": test_data["label"]
    })
    
    # Combine into one DataFrame
    # We'll do our own train/val/test split later
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Get label names from the dataset
    label_names = train_data.features["label"].names
    full_df["label_name"] = full_df["label"].map(lambda x: label_names[x])
    
    # Save to CSV
    os.makedirs("data/raw", exist_ok=True)
    full_df.to_csv("data/raw/banking77.csv", index=False)
    
    # Print statistics
    print(f"\nBanking77 Statistics:")
    print(f"  Total samples: {len(full_df)}")
    print(f"  Number of categories: {full_df['label'].nunique()}")
    print(f"  Average query length: {full_df['text'].str.split().str.len().mean():.1f} words")
    print(f"\n  Sample queries:")
    for i in range(5):
        print(f"    [{full_df.iloc[i]['label_name']}] {full_df.iloc[i]['text']}")
    
    return full_df


def download_clinc150():
    """
    Download CLINC150 dataset.
    Contains 23,700 queries across 150 intents + 1,200 out-of-scope queries.
    We primarily want the out-of-scope queries for our 'General Inquiry' category.
    """
    print("\n" + "=" * 60)
    print("Downloading CLINC150 dataset...")
    print("=" * 60)
    
    # Load the 'plus' config which includes out-of-scope samples
    dataset = load_dataset("clinc_oos", "plus", trust_remote_code=True)
    
    # Combine all splits
    all_texts = []
    all_labels = []
    all_intents = []
    
    # Get intent names
    intent_names = dataset["train"].features["intent"].names
    
    for split_name in ["train", "validation", "test"]:
        split_data = dataset[split_name]
        all_texts.extend(split_data["text"])
        all_labels.extend(split_data["intent"])
        all_intents.extend([intent_names[i] for i in split_data["intent"]])
    
    full_df = pd.DataFrame({
        "text": all_texts,
        "label": all_labels,
        "intent_name": all_intents
    })
    
    # Save to CSV
    full_df.to_csv("data/raw/clinc150.csv", index=False)
    
    # Separate out-of-scope queries
    oos_df = full_df[full_df["intent_name"] == "oos"]
    in_scope_df = full_df[full_df["intent_name"] != "oos"]
    
    # Print statistics
    print(f"\nCLINC150 Statistics:")
    print(f"  Total samples: {len(full_df)}")
    print(f"  In-scope samples: {len(in_scope_df)}")
    print(f"  Out-of-scope samples: {len(oos_df)}")
    print(f"  Number of intents: {full_df['intent_name'].nunique()}")
    print(f"  Average query length: {full_df['text'].str.split().str.len().mean():.1f} words")
    print(f"\n  Sample OUT-OF-SCOPE queries (these go to 'General Inquiry'):")
    for i, row in oos_df.head(5).iterrows():
        print(f"    {row['text']}")
    
    return full_df


if __name__ == "__main__":
    print("SmartTicket - Dataset Download")
    print("=" * 60)
    
    banking_df = download_banking77()
    clinc_df = download_clinc150()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved:")
    print(f"  data/raw/banking77.csv  ({len(banking_df)} rows)")
    print(f"  data/raw/clinc150.csv   ({len(clinc_df)} rows)")