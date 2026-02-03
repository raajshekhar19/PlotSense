"""
FastAPI Application for PlotSense Movie Search.
Main entry point with API endpoints and health checks.
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import API_HOST, API_PORT
from logger import get_logger
from models import SearchRequest, SearchResponse, HealthResponse, ServiceHealthResponse

logger = get_logger(__name__)


# =====================
# Lifespan Management
# =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("=" * 50)
    logger.info("PlotSense Backend Starting...")
    logger.info("=" * 50)
    
    # Initialize services on startup
    try:
        # Import services to trigger initialization
        from services.llm_service import llm_service
        from services.neo4j_service import neo4j_service
        from services.faiss_service import faiss_service
        from services.tavily_service import tavily_service
        from services.reranker_service import reranker_service
        
        logger.info("All services initialized successfully")
        
        # Initialize workflow
        from graph import get_workflow
        get_workflow()
        
        logger.info("Workflow initialized successfully")
        logger.info("=" * 50)
        logger.info(f"Server ready at http://{API_HOST}:{API_PORT}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("PlotSense Backend Shutting Down...")


# =====================
# FastAPI Application
# =====================

app = FastAPI(
    title="PlotSense Movie Search API",
    description="A hybrid movie search API using LangGraph, Neo4j, and FAISS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================
# Health Check Endpoints
# =====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health/services", response_model=ServiceHealthResponse, tags=["Health"])
async def services_health_check():
    """Detailed health check for all services."""
    logger.info("Services health check requested")
    
    services_status = {}
    
    try:
        from services.llm_service import llm_service
        services_status["llm"] = "healthy" if llm_service.is_healthy() else "unhealthy"
    except Exception as e:
        logger.error(f"LLM service health check error: {e}")
        services_status["llm"] = "error"
    
    try:
        from services.neo4j_service import neo4j_service
        services_status["neo4j"] = "healthy" if neo4j_service.is_healthy() else "unhealthy"
    except Exception as e:
        logger.error(f"Neo4j service health check error: {e}")
        services_status["neo4j"] = "error"
    
    try:
        from services.faiss_service import faiss_service
        services_status["faiss"] = "healthy" if faiss_service.is_healthy() else "unhealthy"
    except Exception as e:
        logger.error(f"FAISS service health check error: {e}")
        services_status["faiss"] = "error"
    
    try:
        from services.tavily_service import tavily_service
        services_status["tavily"] = "healthy" if tavily_service.is_healthy() else "unhealthy"
    except Exception as e:
        logger.error(f"Tavily service health check error: {e}")
        services_status["tavily"] = "error"
    
    try:
        from services.reranker_service import reranker_service
        services_status["reranker"] = "healthy" if reranker_service.is_healthy() else "unhealthy"
    except Exception as e:
        logger.error(f"Reranker service health check error: {e}")
        services_status["reranker"] = "error"
    
    overall_status = "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"
    
    logger.info(f"Services health: {services_status}")
    
    return ServiceHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status
    )


# =====================
# Search Endpoint
# =====================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_movies(request: SearchRequest):
    """
    Search for movies based on user query.
    
    Supports:
    - Plot descriptions: "a movie where a man grows potatoes on Mars"
    - Movie names: "movies like Inception"
    - Complex queries: "comedy movies with Tom Hanks from the 90s"
    """
    logger.info("=" * 50)
    logger.info(f"Search request received: {request.query}")
    logger.info("=" * 50)
    
    try:
        from graph import get_workflow
        
        workflow = get_workflow()
        
        # Invoke the workflow
        result = workflow.invoke({"query": request.query})
        
        logger.info(f"Search completed successfully")
        logger.info(f"Intent: {result.get('intent')}")
        logger.info(f"Movie Name: {result.get('movie_name')}")
        logger.info(f"Answer length: {len(result.get('final_answer', ''))}")
        
        return SearchResponse(
            query=request.query,
            intent=result.get("intent"),
            movie_name=result.get("movie_name"),
            answer=result.get("final_answer", "No answer generated"),
            kg_movies=result.get("kg_movies")
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================
# Root Endpoint
# =====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PlotSense Movie Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# =====================
# Run Server
# =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
