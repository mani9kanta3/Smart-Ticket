"""
SmartTicket - Prediction Engine
Loads the ONNX model and handles inference for both
single and batch predictions.
"""

import os
import json
import time
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizer


# Category → Team routing map
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
    """
    Loads the ONNX model and tokenizer once,
    then provides fast predictions.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_mappings = None
        self.max_length = 64
        self.loaded = False
    
    def load(self):
        """Load model, tokenizer, and label mappings."""
        print("Loading SmartTicket model...")
        
        # Load ONNX model
        onnx_path = "models/smartticket_bert.onnx"
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
        self.model = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        print(f"  ONNX model loaded: {onnx_path}")
        
        # Load tokenizer
        tokenizer_path = "models/bert_finetuned"
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        print(f"  Tokenizer loaded: {tokenizer_path}")
        
        # Load label mappings
        with open("data/processed/label_mappings.json", "r") as f:
            self.label_mappings = json.load(f)
        print(f"  Label mappings loaded")
        
        self.loaded = True
        print("SmartTicket model ready!")
    
    def predict(self, text: str, confidence_threshold: float = 0.85) -> dict:
        """
        Classify a single ticket with confidence-based human fallback.
        
        If model confidence is below threshold, flags for human review
        instead of forcing a low-confidence prediction.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        start_time = time.time()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        
        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)
        
        # Run inference
        outputs = self.model.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        
        cat_logits = outputs[0][0]
        pri_logits = outputs[1][0]
        
        # Softmax to get probabilities
        cat_probs = self._softmax(cat_logits)
        pri_probs = self._softmax(pri_logits)
        
        # Get predictions
        cat_id = int(np.argmax(cat_probs))
        pri_id = int(np.argmax(pri_probs))
        
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
        """Classify multiple tickets."""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    @staticmethod
    def _softmax(x):
        """Convert raw logits to probabilities (0-1, sum to 1)."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


# Global predictor instance (loaded once, shared across requests)
predictor = SmartTicketPredictor()