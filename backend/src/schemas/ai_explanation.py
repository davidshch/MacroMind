"""
Pydantic schemas for AI Explanation / Insight Navigator.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from datetime import datetime

class AIExplanationRequestContextAsset(BaseModel):
    type: Literal["asset"] = "asset"
    symbol: str
    # Optional: specific date range or event to focus on for the asset
    related_event_id: Optional[int] = None
    lookback_days: Optional[int] = 7

class AIExplanationRequestContextEvent(BaseModel):
    type: Literal["event"] = "event"
    event_id: int
    # Optional: specific asset to focus on in relation to the event
    related_symbol: Optional[str] = None 

class AIExplanationRequestContextGeneral(BaseModel):
    type: Literal["general"] = "general"
    # For general market queries, context might be less specific
    market_focus: Optional[str] = None # e.g., "US Equities", "Crypto"

# In a more advanced version, context could be a discriminated union
# For MVP, let's keep it simple or allow flexible dict

class AIExplanationRequest(BaseModel):
    query: str = Field(..., description="The user's question for the AI to explain.")
    # Context can be flexible for MVP, specific models above show future direction
    context_params: Optional[Dict[str, Any]] = Field(None, description="Parameters to help gather context, e.g., {'symbol': 'AAPL', 'event_id': 123}")

class AIExplanationResponse(BaseModel):
    query: str
    explanation: str
    confidence_score: Optional[float] = None # LLM might provide this, or we estimate
    supporting_data_summary: Optional[Dict[str, Any]] = None # Brief summary of data used for context
    generated_at: datetime
    # For future VIP: contributing_factors, historical_precedents etc.
