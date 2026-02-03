"""
Pydantic models and TypedDict definitions for PlotSense backend.
"""
from typing import TypedDict, Optional, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


# =====================
# Workflow State
# =====================

class MovieState(TypedDict):
    """State passed through the LangGraph workflow."""
    query: str
    intent: Optional[str]  # plot | movie_name | query_search | invalid
    movie_name: Optional[str]
    movie_exists: Optional[bool]
    base_plot: Optional[str]
    kg_movies: Optional[List[str]]
    final_docs: Optional[List[Any]]  # List[Document]
    final_answer: Optional[str]


# =====================
# LLM Structured Outputs
# =====================

class IntentClassification(BaseModel):
    """The intent category of the user's movie query."""
    intent: str = Field(
        description="Must be one of: 'plot' (describing a story), 'movie_name' (specific movie title), 'query_search' (complex query), or 'invalid' (gibberish, off-topic, or random text)."
    )
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0")


class MovieFilters(BaseModel):
    """Extracted entities for advanced Knowledge Graph querying."""
    actor: Optional[str] = Field(None, description="Name of the actor.")
    director: Optional[str] = Field(None, description="Name of the director.")
    genre: Optional[str] = Field(None, description="The movie genre.")
    keywords: Optional[List[str]] = Field(
        default_factory=list,
        description="Key plot points or themes (e.g., 'time travel', 'sinking ship')."
    )


# =====================
# API Request/Response Models
# =====================

class SearchRequest(BaseModel):
    """Request model for movie search endpoint."""
    query: str = Field(..., description="User's movie search query", min_length=1)


class SearchResponse(BaseModel):
    """Response model for movie search endpoint."""
    query: str
    intent: Optional[str] = None
    movie_name: Optional[str] = None
    answer: str
    kg_movies: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    timestamp: str


class ServiceHealthResponse(BaseModel):
    """Response model for detailed service health check."""
    status: str
    timestamp: str
    services: dict
