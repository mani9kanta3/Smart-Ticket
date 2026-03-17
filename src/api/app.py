"""
SmartTicket - FastAPI Application
REST API for ticket classification with single and batch endpoints.
"""

import os
import io
import time
import csv
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    TicketRequest,
    TicketResponse,
    BatchResponse,
    HealthResponse,
    ModelComparisonResponse,
)
from src.api.predict import predictor


# Track server start time for uptime calculation
SERVER_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model when server starts."""
    predictor.load()
    yield
    # Cleanup on shutdown (nothing needed)


app = FastAPI(
    title="SmartTicket API",
    description="Intelligent Customer Support Ticket Classifier & Priority Router",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS (so Streamlit dashboard can call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Endpoints
# ============================================================

@app.post("/classify", response_model=TicketResponse)
async def classify_ticket(request: TicketRequest):
    """
    Classify a single support ticket.
    
    Returns category, priority, confidence, and routing team.
    """
    try:
        result = predictor.predict(request.text)
        return TicketResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_classify")
async def batch_classify(file: UploadFile = File(...)):
    """
    Classify tickets from a CSV file.
    
    Upload a CSV with a 'text' column. Returns a CSV with
    added columns: category, priority, confidence, routing_team.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if "text" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must have a 'text' column"
            )
        
        # Classify each ticket
        start_time = time.time()
        results = predictor.predict_batch(df["text"].tolist())
        total_time = (time.time() - start_time) * 1000
        
        # Add predictions to dataframe
        df["category"] = [r["category"] for r in results]
        df["priority"] = [r["priority"] for r in results]
        df["confidence"] = [r["category_confidence"] for r in results]
        df["routing_team"] = [r["routing_team"] for r in results]
        
        # Return as downloadable CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=classified_{file.filename}",
                "X-Total-Tickets": str(len(df)),
                "X-Total-Time-Ms": f"{total_time:.2f}",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    return HealthResponse(
        status="healthy" if predictor.loaded else "model not loaded",
        model_loaded=predictor.loaded,
        model_type="DistilBERT + ONNX Runtime",
        uptime_seconds=round(time.time() - SERVER_START_TIME, 1),
    )


@app.get("/models/compare")
async def model_comparison():
    """Return comparison metrics for all 4 models."""
    import json
    comparison_path = "models/model_comparison.json"
    if not os.path.exists(comparison_path):
        raise HTTPException(status_code=404, detail="Comparison data not found")
    
    with open(comparison_path, "r") as f:
        data = json.load(f)
    
    return data