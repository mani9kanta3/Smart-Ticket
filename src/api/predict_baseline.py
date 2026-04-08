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

    def predict(self, text: str, confidence_threshold: float = 0.85) -> dict:
        """
        Classify a single ticket with confidence-based human fallback.
        
        If model confidence is below threshold, flags for human review
        instead of forcing a low-confidence prediction. This is critical
        for production ML systems — the model should know what it doesn't know.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start_time = time.time()

        X = self.tfidf.transform([text])

        cat_id = int(self.cat_model.predict(X)[0])
        pri_id = int(self.pri_model.predict(X)[0])

        cat_probs = self.cat_model.predict_proba(X)[0]
        pri_probs = self.pri_model.predict_proba(X)[0]

        cat_confidence = float(cat_probs[cat_id])
        pri_confidence = float(pri_probs[pri_id])

        cat_name = self.label_mappings["id_to_category"][str(cat_id)]
        pri_name = self.label_mappings["id_to_priority"][str(pri_id)]

        inference_time = (time.time() - start_time) * 1000

        # Confidence-based routing decision
        if cat_confidence < confidence_threshold:
            review_status = "needs_human_review"
            review_reason = f"Low confidence ({cat_confidence:.1%} < {confidence_threshold:.0%} threshold)"
            routing_team = "Human Review Queue"
        else:
            review_status = "auto_routed"
            review_reason = None
            routing_team = ROUTING_MAP.get(cat_name, "General Support Team")

        return {
            "text": text,
            "category": cat_name,
            "category_id": cat_id,
            "category_confidence": round(cat_confidence, 4),
            "priority": pri_name,
            "priority_id": pri_id,
            "priority_confidence": round(pri_confidence, 4),
            "routing_team": routing_team,
            "review_status": review_status,
            "review_reason": review_reason,
            "inference_time_ms": round(inference_time, 2),
        }

    def predict_batch(self, texts: list) -> list:
        return [self.predict(text) for text in texts]


predictor = SmartTicketPredictor()