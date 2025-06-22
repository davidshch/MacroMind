from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Insight(BaseModel):
    """A single piece of actionable insight."""
    title: str = Field(..., description="The headline of the insight, e.g., 'High Volatility Warning'.")
    description: str = Field(..., description="A detailed explanation of the insight and its implications.")
    confidence: float = Field(..., ge=0, le=1, description="The model's confidence in this insight.")
    category: str = Field(..., description="Category of the insight, e.g., 'Volatility', 'Sentiment', 'Opportunity'.")

class AIAnalystResponse(BaseModel):
    """The complete response from the AI Analyst service."""
    symbol: str = Field(..., description="The symbol for which insights were generated.")
    generated_at: datetime = Field(default_factory=datetime.now, description="Timestamp of when the insights were generated.")
    overall_outlook: str = Field(..., description="A brief, high-level outlook, e.g., 'Bullish but Volatile'.")
    summary: str = Field(..., description="A short paragraph summarizing the key takeaways.")
    insights: List[Insight] = Field(..., description="A list of specific, actionable insights.") 