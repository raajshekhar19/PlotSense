"""
FastAPI Application for PlotSense Movie Search.
Main entry point with API endpoints and health checks.
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from config import API_HOST, API_PORT, LANGSMITH_TRACING, LANGSMITH_PROJECT, LANGSMITH_ENDPOINT
from logger import get_logger
from models import SearchRequest, SearchResponse, HealthResponse, ServiceHealthResponse

logger = get_logger(__name__)


# =====================
# Rate Limiter Setup
# =====================

# IPs that bypass all rate limits (localhost for dev & health-check scripts)
_WHITELISTED_IPS = {"127.0.0.1", "::1"}


def _get_real_ip(request: Request) -> str:
    """
    Extract the *real* client IP, respecting X-Forwarded-For when behind a
    reverse proxy (Render, Fly.io, nginx, etc.).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # The first address is the original client
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    return client_ip


def _is_whitelisted(request: Request) -> bool:
    """Return True if the request comes from a whitelisted (localhost) IP."""
    client_ip = _get_real_ip(request)
    return client_ip in _WHITELISTED_IPS


limiter = Limiter(
    key_func=_get_real_ip,
    default_limits=["30/minute"],           # Global fallback: 30 req/min/IP
    storage_uri="memory://",
)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Custom 429 handler that returns a clean JSON body and Retry-After header.
    """
    # Parse the window reset time from the exception details
    retry_after = int(getattr(exc, "retry_after", 60) or 60)
    # SlowAPI stores the detail string; try to extract a sensible retry window
    try:
        # exc.detail often looks like "Rate limit exceeded: 10 per 1 minute"
        # We fall back to 60s if we cannot parse
        window_seconds = 60
        detail = str(getattr(exc, "detail", ""))
        if "minute" in detail:
            window_seconds = 60
        elif "hour" in detail:
            window_seconds = 3600
        elif "second" in detail:
            window_seconds = 1
        retry_after = window_seconds
    except Exception:
        retry_after = 60

    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "message": "You are sending too many requests. Please slow down.",
            "retry_after_seconds": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    """
    Middleware that injects X-RateLimit-* headers into every response
    by reading the state that SlowAPI writes to request.state.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        view_rate_limit = getattr(request.state, "view_rate_limit", None)
        if view_rate_limit:
            # view_rate_limit is a string like "10 per 1 minute"
            try:
                parts = view_rate_limit.split()
                limit_value = parts[0]
                response.headers["X-RateLimit-Limit"] = limit_value
            except Exception:
                pass

        rate_limit_remaining = getattr(request.state, "_rate_limiting_remaining", None)
        if rate_limit_remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_remaining)

        rate_limit_reset = getattr(request.state, "_rate_limiting_reset", None)
        if rate_limit_reset is not None:
            response.headers["X-RateLimit-Reset"] = str(rate_limit_reset)

        return response


# =====================
# Lifespan Management
# =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("=" * 50)
    logger.info("PlotSense Backend Starting...")
    logger.info("=" * 50)
    
    # Initialize services on startup — each service fails independently
    _services = [
        ("llm_service",      "services.llm_service"),
        ("neo4j_service",    "services.neo4j_service"),
        ("faiss_service",    "services.faiss_service"),
        ("tavily_service",   "services.tavily_service"),
        ("reranker_service", "services.reranker_service"),
    ]
    for svc_attr, svc_module in _services:
        try:
            import importlib
            importlib.import_module(svc_module)
            logger.info(f"  ✔ {svc_attr} ready")
        except Exception as e:
            logger.warning(f"  ✘ {svc_attr} failed to init (non-fatal): {e}")
    
    # Initialize workflow
    try:
        from graph import get_workflow
        get_workflow()
        logger.info("Workflow initialized successfully")
    except Exception as e:
        logger.warning(f"Workflow init failed (non-fatal): {e}")
    
    logger.info("=" * 50)
    
    # LangSmith tracing status
    if LANGSMITH_TRACING.lower() == "true":
        logger.info("✅ LangSmith tracing ENABLED")
        logger.info(f"   Project : {LANGSMITH_PROJECT}")
        logger.info(f"   Endpoint: {LANGSMITH_ENDPOINT}")
    else:
        logger.warning("⚠️  LangSmith tracing DISABLED")
    
    logger.info(f"Server ready at http://{API_HOST}:{API_PORT}")
    logger.info("=" * 50)
    
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

# --- Rate limiter integration ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(RateLimitHeaderMiddleware)

# CORS middleware (must be outermost to set CORS headers on 429 responses too)
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
@limiter.limit("60/minute", exempt_when=_is_whitelisted)
async def health_check(request: Request):
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


import asyncio
import urllib.request
import json
import os

def fetch_poster_sync(title: str) -> str:
    """Fetch movie poster using IMDb's public suggestion API synchronously."""
    if not title: return ""
    
    import re
    # Clean string: lowercase, replace space with _, keep only alphanum and _
    formatted = re.sub(r'[^a-zA-Z0-9_\-]', '', title.lower().replace(' ', '_'))
    if not formatted: return ""
    
    first_letter = formatted[0]
    url = f"https://v3.sg.media-imdb.com/suggestion/{first_letter}/{formatted}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as res:
            data = json.loads(res.read().decode('utf-8'))
            for item in data.get("d", []):
                # Return the first item that has an image
                if "i" in item and "imageUrl" in item["i"]:
                    # Keep original high res image rather than _V1_UX68_ scale defaults
                    img_url = item["i"]["imageUrl"]
                    return img_url.replace("._V1_.jpg", "._V1_SX600_.jpg")
            return ""
    except Exception as e:
        logger.warning(f"Failed to fetch poster for {title}: {e}")
        return ""


# =====================
# Search Endpoint
# =====================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
@limiter.limit("10/minute", exempt_when=_is_whitelisted)
async def search_movies(request: Request, search_request: SearchRequest):
    """
    Search for movies based on user query.
    
    Supports:
    - Plot descriptions: "a movie where a man grows potatoes on Mars"
    - Movie names: "movies like Inception"
    - Complex queries: "comedy movies with Tom Hanks from the 90s"
    """
    logger.info("=" * 50)
    logger.info(f"Search request received: {search_request.query}")
    logger.info("=" * 50)
    
    try:
        from graph import get_workflow
        
        workflow = get_workflow()
        
        # Invoke the workflow
        result = workflow.invoke({"query": search_request.query})
        
        logger.info(f"Search completed successfully")
        logger.info(f"Intent: {result.get('intent')}")
        logger.info(f"Movie Name: {result.get('movie_name')}")
        logger.info(f"Answer length: {len(result.get('final_answer', ''))}")
        
        # Populate rich metadata objects for the frontend UI cards
        movies_list = []
        if result.get("final_docs"):
            for d in result["final_docs"]:
                plot_preview = d.page_content.strip()
                    
                movies_list.append({
                    "title": d.metadata.get("title", "Unknown Title"),
                    "snippet": plot_preview,
                    "source": d.metadata.get("source", "PlotSense DB"),
                    "year": d.metadata.get("year", ""),
                    "director": d.metadata.get("director", "")
                })
        elif result.get("kg_movies"):
            movies_list = result["kg_movies"]
            
        # Concurrently fetch posters
        if movies_list:
            async def get_poster_for_idx(idx: int, title_str: str, stagger_ms: float):
                if title_str:
                    await asyncio.sleep(stagger_ms)
                    poster_url = await asyncio.to_thread(fetch_poster_sync, title_str)
                    if poster_url and isinstance(movies_list[idx], dict):
                        movies_list[idx]["posterUrl"] = poster_url
            
            tasks = []
            for i, m in enumerate(movies_list):
                if isinstance(m, dict):
                    t = m.get("title", "")
                elif isinstance(m, str):
                    t = m
                    movies_list[i] = {"title": m, "snippet": "", "source": "PlotSense DB", "year": "", "director": ""}
                else:
                    t = ""
                
                # Stagger requests by 200ms per index to avoid rate limits
                tasks.append(get_poster_for_idx(i, t, i * 0.2))
            
            if tasks:
                await asyncio.gather(*tasks)
            
        return SearchResponse(
            query=search_request.query,
            intent=result.get("intent"),
            movie_name=result.get("movie_name"),
            answer=result.get("final_answer", "No answer generated"),
            kg_movies=movies_list,
            needs_clarification=result.get("needs_clarification", False),
            clarification_question=result.get("clarification_question"),
            search_status=result.get("search_status", "success"),
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
