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
    
    # Fallback to Tavily if FAISS vector index is missing or returns absolutely nothing
    if not final_docs:
        logger.warning("FAISS returned 0 results. Falling back to Tavily Web Search...")
        try:
            from services.tavily_service import tavily_service
            search_query = f"{query_text} movie title"
            raw_response = tavily_service.search_tool.invoke({"query": search_query})
            
            results = raw_response.get("results", []) if isinstance(raw_response, dict) else []
            
            from langchain_core.documents import Document
            for r in results[:3]:
                if isinstance(r, dict) and r.get("content"):
                    final_docs.append(Document(
                        page_content=r["content"],
                        metadata={"title": r.get("title", "Web Result"), "source": "Tavily Web Search", "url": r.get("url", "")}
                    ))
            
            if final_docs:
                logger.info(f"Retrieved {len(final_docs)} fallback documents from Tavily.")
        except Exception as e:
            logger.error(f"Tavily fallback failed: {e}")
    
    logger.info(f"Aggregated to {len(final_docs)} final documents.")
    
    return {"final_docs": final_docs}
