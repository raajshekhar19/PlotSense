"""
Hybrid Rerank Node for PlotSense backend.
Combines semantic and symbolic scores for reranking.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models import MovieState
from services.reranker_service import reranker_service
from logger import get_logger

logger = get_logger(__name__)


def hybrid_rerank(state: MovieState) -> dict:
    """
    Perform hybrid reranking using cross-encoder and KG boost.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with reranked final documents
    """
    logger.info("--- Node: Hybrid Rerank (Cross-Encoder) ---")
    
    docs = state.get("final_docs", [])
    
    if not docs:
        logger.warning("No documents to rerank")
        return {"final_docs": []}
    
    kg_movies = state.get("kg_movies", [])
    query = state["query"]
    
    top_docs = reranker_service.rerank(query, docs, kg_movies)
    
    return {"final_docs": top_docs}
