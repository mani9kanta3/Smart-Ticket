"""
SmartTicket - Lightweight Predictor for Cloud Deployment
Uses TF-IDF + Logistic Regression (5MB) instead of ONNX (254MB)
for fast cloud deployment.
"""

import os
import json
import time
import joblib
import numpy as np


ROUTING_MAP = {
    "Account Access": "Security & Authentication Team",
    "Account Management": "Account Services Team",
    "Balance & Statement": "Account Services Team",
    "Card Services": "Card Operations Team",
    "Fees & Charges": "Billing & Fees Team",
    "General Inquiry": "General Support Team",
    "Payment Issues": "Payments Team",
    "Refund & Dispute": "Disputes & Resolution Team",
    "Technical Issues": "Technical Support Team",
    "Transfer & Transaction": "Transfers Team",
}


class SmartTicketPredictor:
    def __init__(self):
        self.tfidf = None
        self.cat_model = None
        self.pri_model = None
        self.label_mappings = None
        self.loaded = False

    def load(self):
        print("Loading SmartTicket baseline model...")
        self.tfidf = joblib.load("models/tfidf_vectorizer.joblib")
        self.cat_model = joblib.load("models/baseline_category_model.joblib")
        self.pri_model = joblib.load("models/baseline_priority_model.joblib")

        with open("data/processed/label_mappings.json", "r") as f:
            self.label_mappings = json.load(f)

        self.loaded = True
        print("SmartTicket baseline model ready!")

    def predict(self, text: str) -> dict:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        X = self.tfidf.transform([text])

        cat_id = int(self.cat_model.predict(X)[0])
        pri_id = int(self.pri_model.predict(X)[0])

        cat_probs = self.cat_model.predict_proba(X)[0]
        pri_probs = self.pri_model.predict_proba(X)[0]

        cat_name = self.label_mappings["id_to_category"][str(cat_id)]
        pri_name = self.label_mappings["id_to_priority"][str(pri_id)]

        inference_time = (time.time() - start_time) * 1000

        return {
            "text": text,
            "category": cat_name,
            "category_id": cat_id,
            "category_confidence": round(float(cat_probs[cat_id]), 4),
            "priority": pri_name,
            "priority_id": pri_id,
            "priority_confidence": round(float(pri_probs[pri_id]), 4),
            "routing_team": ROUTING_MAP.get(cat_name, "General Support Team"),
            "inference_time_ms": round(inference_time, 2),
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(text) for text in texts]


predictor = SmartTicketPredictor()