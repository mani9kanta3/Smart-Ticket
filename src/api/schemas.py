"""
SmartTicket - API Request/Response Schemas
Defines the exact shape of data going in and out of the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class TicketRequest(BaseModel):
    """Single ticket classification request."""
    text: str = Field(..., min_length=1, max_length=1000,
                      description="Customer support ticket text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "My card payment was declined and I need help urgently"
            }
        }


class TicketResponse(BaseModel):
    """Single ticket classification response."""
    text: str
    category: str
    category_id: int
    category_confidence: float
    priority: str
    priority_id: int
    priority_confidence: float
    routing_team: str
    inference_time_ms: float


class BatchResponse(BaseModel):
    """Batch classification response."""
    total_tickets: int
    results: List[TicketResponse]
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    model_loaded: bool
    model_type: str
    uptime_seconds: float


class ModelComparisonResponse(BaseModel):
    """Model comparison metrics."""
    models: dict