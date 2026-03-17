"""
SmartTicket - Data Preparation
Consolidates 77 Banking77 categories into 10 business groups,
adds CLINC150 out-of-scope queries, creates priority labels,
and performs train/val/test split.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import yaml

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


# ============================================================
# CATEGORY CONSOLIDATION MAPPING
# ============================================================
# Each of the 77 Banking77 labels maps to one of 10 business groups.
# The logic: labels that would be handled by the SAME support team
# get grouped together.

CATEGORY_MAPPING = {
    # Category 0: Payment Issues
    # All card payment problems — same team handles these
    "card_payment_not_recognised": "Payment Issues",
    "card_payment_wrong_exchange_rate": "Payment Issues",
    "card_payment_fee_charged": "Payment Issues",
    "declined_card_payment": "Payment Issues",
    "reverted_card_payment?": "Payment Issues",
    "pending_card_payment": "Payment Issues",
    "card_not_working": "Payment Issues",
    "card_payment_wrong_amount": "Payment Issues",  # custom mapping added
    
    # Category 1: Transfer & Transaction
    # Money movement between accounts
    "transfer_timing": "Transfer & Transaction",
    "transfer_not_received_by_recipient": "Transfer & Transaction",
    "failed_transfer": "Transfer & Transaction",
    "pending_transfer": "Transfer & Transaction",
    "transfer_fee_charged": "Transfer & Transaction",
    "wrong_amount_of_cash_received": "Transfer & Transaction",
    "receiving_money": "Transfer & Transaction",
    "transfer_into_account": "Transfer & Transaction",
    
    # Category 2: Account Access
    # Login, verification, security — account access team
    "passcode_forgotten": "Account Access",
    "compromised_card": "Account Access",
    "card_not_active": "Account Access",
    "pin_blocked": "Account Access",
    "authenticate_via_phone": "Account Access",  # custom mapping added
    "verify_my_identity": "Account Access",
    "unable_to_verify_identity": "Account Access",
    "why_verify_identity": "Account Access",
    
    # Category 3: Card Services
    # Physical/virtual card lifecycle — card services team
    "lost_or_stolen_card": "Card Services",
    "card_arrival": "Card Services",
    "card_delivery_estimate": "Card Services",
    "card_linking": "Card Services",
    "getting_virtual_card": "Card Services",
    "contactless_not_working": "Card Services",
    "card_about_to_expire": "Card Services",
    "get_physical_card": "Card Services",
    "virtual_card_not_working": "Card Services",
    "card_swallowed": "Card Services",
    "activate_my_card": "Card Services",
    "get_disposable_virtual_card": "Card Services",
    "card_acceptance": "Card Services",
    
    # Category 4: Refund & Dispute
    # Money back, charge disputes — disputes team
    "request_refund": "Refund & Dispute",
    "Refund_not_showing_up": "Refund & Dispute",
    "direct_debit_payment_not_recognised": "Refund & Dispute",
    "transaction_charged_twice": "Refund & Dispute",
    "cancel_transfer": "Refund & Dispute",
    "declined_cash_withdrawal": "Refund & Dispute",
    
    # Category 5: Balance & Statement
    # Checking money, statements — informational, self-service
    "balance_not_updated_after_bank_transfer": "Balance & Statement",
    "balance_not_updated_after_cheque_or_cash_deposit": "Balance & Statement",
    "cash_withdrawal_not_recognised": "Balance & Statement",
    "pending_cash_withdrawal": "Balance & Statement",
    "wrong_exchange_rate_for_cash_withdrawal": "Balance & Statement",
    "cash_withdrawal_charge": "Balance & Statement",
    
    # Category 6: Fees & Charges
    # Fee inquiries — billing team
    "extra_charge_on_statement": "Fees & Charges",
    "exchange_rate": "Fees & Charges",
    "exchange_charge": "Fees & Charges",
    "exchange_via_app": "Fees & Charges",
    "beneficiary_not_allowed": "Fees & Charges",
    "fiat_currency_support": "Fees & Charges",
    
    # Category 7: Account Management
    # Account lifecycle — account management team
    "terminate_account": "Account Management",
    "change_pin": "Account Management",
    "supported_cards_and_currencies": "Account Management",
    "age_limit": "Account Management",
    "country_support": "Account Management",
    "automatic_top_up": "Account Management",
    "top_up_failed": "Account Management",
    "top_up_limits": "Account Management",
    "top_up_reverted": "Account Management",
    "top_up_by_bank_transfer_charge": "Account Management",
    "top_up_by_card_charge": "Account Management",
    "topping_up_by_card": "Account Management",
    "pending_top_up": "Account Management",
    
    # Category 8: Technical Issues
    # App errors, system problems — technical support team
    "apple_pay_or_google_pay": "Technical Issues",
    "atm_support": "Technical Issues",
    "visa_or_mastercard": "Technical Issues",
    
    # Category 9: General Inquiry
    # General questions, complaints, anything that doesn't fit above
    "edit_personal_details": "General Inquiry",
    "order_physical_card": "General Inquiry",
}


def consolidate_categories():
    """
    Map 77 Banking77 labels to 10 business categories.
    Add CLINC150 out-of-scope queries as 'General Inquiry'.
    """
    print("=" * 60)
    print("STEP 1: Consolidating 77 categories into 10 groups")
    print("=" * 60)
    
    # Load raw data
    banking = pd.read_csv("data/raw/banking77.csv")
    clinc = pd.read_csv("data/raw/clinc150.csv")
    
    # Map Banking77 labels to consolidated categories
    banking["category"] = banking["label_name"].map(CATEGORY_MAPPING)
    
    # Check for unmapped labels
    unmapped = banking[banking["category"].isnull()]["label_name"].unique()
    if len(unmapped) > 0:
        print(f"\n⚠️  WARNING: {len(unmapped)} unmapped labels found:")
        for label in unmapped:
            print(f"    - {label}")
        print("\n  Assigning unmapped labels to 'General Inquiry'")
        banking["category"] = banking["category"].fillna("General Inquiry")
    else:
        print("\n✅ All 77 labels mapped successfully!")
    
    # Get CLINC150 out-of-scope queries
    clinc_oos = clinc[clinc["intent_name"] == "oos"][["text"]].copy()
    clinc_oos["category"] = "General Inquiry"
    clinc_oos["label"] = -1  # placeholder
    clinc_oos["label_name"] = "oos"
    
    print(f"\n  Banking77 samples: {len(banking)}")
    print(f"  CLINC150 OOS samples added: {len(clinc_oos)}")
    
    # Combine
    consolidated = pd.concat([
        banking[["text", "label_name", "category"]],
        clinc_oos[["text", "label_name", "category"]]
    ], ignore_index=True)
    
    print(f"  Total combined samples: {len(consolidated)}")
    
    # Show distribution
    print(f"\n  Category Distribution:")
    print(f"  {'Category':<25s} {'Count':>6s} {'Percentage':>10s}")
    print(f"  {'-'*45}")
    cat_counts = consolidated["category"].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(consolidated) * 100
        print(f"  {cat:<25s} {count:>6d} {pct:>9.1f}%")
    
    # Create numeric labels for the 10 categories
    category_list = sorted(consolidated["category"].unique())
    cat_to_id = {cat: idx for idx, cat in enumerate(category_list)}
    consolidated["category_id"] = consolidated["category"].map(cat_to_id)
    
    print(f"\n  Category ID mapping:")
    for cat, idx in sorted(cat_to_id.items(), key=lambda x: x[1]):
        print(f"    {idx}: {cat}")
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", 10)
    cat_counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Consolidated 10-Category Distribution")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add count labels on bars
    for i, (cat, count) in enumerate(cat_counts.items()):
        ax.text(i, count + 30, str(count), ha="center", fontsize=9, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("reports/figures/consolidated_category_distribution.png", dpi=150)
    plt.close()
    print("\n  Saved: reports/figures/consolidated_category_distribution.png")
    
    return consolidated, cat_to_id


def create_priority_labels(df):
    """
    Assign priority labels (P0-P3) based on category and keywords.
    
    P0 - Critical: Security threats, locked accounts
    P1 - High: Failed transactions, money not received
    P2 - Medium: Refunds, card requests, changes
    P3 - Low: Informational queries, general questions
    """
    print("\n" + "=" * 60)
    print("STEP 2: Creating Priority Labels")
    print("=" * 60)
    
    # Step 1: Assign base priority from category
    category_priority = {
        "Account Access": "P1-High",        # Can't access account is urgent
        "Payment Issues": "P1-High",         # Payment problems are urgent
        "Transfer & Transaction": "P1-High", # Money movement issues
        "Card Services": "P2-Medium",        # Card requests take time
        "Refund & Dispute": "P2-Medium",     # Refunds need investigation
        "Balance & Statement": "P2-Medium",  # Balance queries
        "Technical Issues": "P2-Medium",     # App/system issues
        "Fees & Charges": "P3-Low",          # Fee inquiries
        "Account Management": "P3-Low",      # Account admin tasks
        "General Inquiry": "P3-Low",         # General questions
    }
    
    df["priority"] = df["category"].map(category_priority)
    
    # Step 2: Override with keyword-based rules
    # Some queries within a category are more urgent than others
    text_lower = df["text"].str.lower()
    
    # P0-Critical: Security emergencies
    p0_keywords = [
        "stolen", "hacked", "unauthorized", "fraud", "scam",
        "locked out", "compromised", "security breach", "identity theft",
        "someone else", "not me", "didn't authorize"
    ]
    p0_mask = text_lower.apply(lambda x: any(kw in x for kw in p0_keywords))
    df.loc[p0_mask, "priority"] = "P0-Critical"
    
    # P1-High: Money at risk
    p1_keywords = [
        "failed", "not working", "not received", "wrong amount",
        "charged twice", "double charged", "missing money",
        "can't pay", "declined", "urgent", "immediately"
    ]
    p1_mask = text_lower.apply(lambda x: any(kw in x for kw in p1_keywords))
    # Only upgrade, don't downgrade P0 to P1
    df.loc[p1_mask & (df["priority"] != "P0-Critical"), "priority"] = "P1-High"
    
    # Create numeric priority
    priority_to_id = {
        "P0-Critical": 0,
        "P1-High": 1,
        "P2-Medium": 2,
        "P3-Low": 3
    }
    df["priority_id"] = df["priority"].map(priority_to_id)
    
    # Show distribution
    print(f"\n  Priority Distribution:")
    print(f"  {'Priority':<15s} {'Count':>6s} {'Percentage':>10s}")
    print(f"  {'-'*35}")
    pri_counts = df["priority"].value_counts().sort_index()
    for pri, count in pri_counts.items():
        pct = count / len(df) * 100
        print(f"  {pri:<15s} {count:>6d} {pct:>9.1f}%")
    
    # Show examples for each priority
    print(f"\n  Sample queries per priority:")
    for pri in ["P0-Critical", "P1-High", "P2-Medium", "P3-Low"]:
        samples = df[df["priority"] == pri]["text"].head(2).tolist()
        print(f"\n  [{pri}]")
        for s in samples:
            print(f"    → {s}")
    
    # Plot priority distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors_pri = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"]
    pri_counts.plot(kind="bar", ax=ax, color=colors_pri)
    ax.set_xlabel("Priority Level")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Priority Distribution")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, (pri, count) in enumerate(pri_counts.items()):
        ax.text(i, count + 30, str(count), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/priority_distribution.png", dpi=150)
    plt.close()
    print("\n  Saved: reports/figures/priority_distribution.png")
    
    return df, priority_to_id


def split_data(df):
    """
    Split into train (80%), validation (10%), test (10%).
    Uses stratified sampling so each split has similar category proportions.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Train / Validation / Test Split")
    print("=" * 60)
    
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config["data"]["random_state"],
        stratify=df["category_id"]
    )
    
    # Second split: temp → 50% val, 50% test (so overall 80/10/10)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=config["data"]["random_state"],
        stratify=temp_df["category_id"]
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"\n  Split sizes:")
    print(f"    Train:      {len(train_df):>6d} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Validation: {len(val_df):>6d} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Test:       {len(test_df):>6d} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    Total:      {len(df):>6d}")
    
    # Verify stratification — category proportions should be similar
    print(f"\n  Category proportions (verifying stratification):")
    print(f"  {'Category':<25s} {'Train':>7s} {'Val':>7s} {'Test':>7s}")
    print(f"  {'-'*50}")
    for cat_id in sorted(df["category_id"].unique()):
        cat_name = df[df["category_id"] == cat_id]["category"].iloc[0]
        train_pct = (train_df["category_id"] == cat_id).mean() * 100
        val_pct = (val_df["category_id"] == cat_id).mean() * 100
        test_pct = (test_df["category_id"] == cat_id).mean() * 100
        print(f"  {cat_name:<25s} {train_pct:>6.1f}% {val_pct:>6.1f}% {test_pct:>6.1f}%")
    
    # Save splits
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print(f"\n  Saved:")
    print(f"    data/processed/train.csv")
    print(f"    data/processed/val.csv")
    print(f"    data/processed/test.csv")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("SmartTicket - Data Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Consolidate categories
    consolidated, cat_to_id = consolidate_categories()
    
    # Step 2: Create priority labels
    consolidated, priority_to_id = create_priority_labels(consolidated)
    
    # Save consolidated data before splitting
    consolidated.to_csv("data/processed/consolidated_data.csv", index=False)
    print(f"\n  Saved: data/processed/consolidated_data.csv ({len(consolidated)} rows)")
    
    # Step 3: Split data
    train_df, val_df, test_df = split_data(consolidated)
    
    # Save mappings for later use
    import json
    mappings = {
        "category_to_id": cat_to_id,
        "id_to_category": {v: k for k, v in cat_to_id.items()},
        "priority_to_id": priority_to_id,
        "id_to_priority": {v: k for k, v in priority_to_id.items()},
    }
    with open("data/processed/label_mappings.json", "w") as f:
        json.dump(mappings, f, indent=2)
    print(f"\n  Saved: data/processed/label_mappings.json")
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\n  10 Categories: {list(cat_to_id.keys())}")
    print(f"  4 Priorities:  {list(priority_to_id.keys())}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")