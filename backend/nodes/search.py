"""
Similarity Search Node for PlotSense backend.
Performs FAISS similarity search with aggregation.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models import MovieState
from services.faiss_service import faiss_service
from logger import get_logger

logger = get_logger(__name__)


def similarity_search(state: MovieState) -> dict:
    """
    Perform aggregated similarity search.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with final documents
    """
    logger.info("--- Node: Similarity Search (aggregated) ---")
    
    query_text = state.get("base_plot") or state["query"]
    
    logger.debug(f"Search query: {query_text[:100]}...")
    
    # Fetch more chunks than needed to get diverse movies
    raw_docs = faiss_service.similarity_search(query_text, k=25)
    
    # Group chunks by Movie Name (Deduplicate)
    unique_movies = {}
    for doc in raw_docs:
        title = doc.metadata.get("title", "Unknown")
        
        if title not in unique_movies:
            unique_movies[title] = {
                "doc": doc,
                "score": 0.0,
                "chunks": [doc.page_content]
            }
        else:
            unique_movies[title]["chunks"].append(doc.page_content)
    
    # Limit to top 5 unique movies
    final_docs = []
    for title, data in list(unique_movies.items())[:5]:
        final_docs.append(data["doc"])
    
    logger.info(f"Aggregated to {len(final_docs)} unique movies.")
    
    return {"final_docs": final_docs}
