"""
Reranking Node for PlotSense backend.
Matches hybrid_search_verbose.ipynb exactly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import CrossEncoder
from models import MovieState
from logger import get_logger

logger = get_logger(__name__)

# Initialize cross-encoder globally (loaded ONCE, not inside function)
try:
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("CrossEncoder reranker loaded successfully")
except Exception as e:
    logger.error(f"Failed to load CrossEncoder: {e}")
    reranker_model = None

def hybrid_rerank(state: MovieState) -> dict:
    """
    Hybrid rerank using Cross-Encoder + KG symbolic boost.
    Matches notebook exactly.
    """
    logger.info("--- Node: Hybrid Rerank (Cross-Encoder) ---")
    
    docs = state.get("final_docs", [])
    if not docs or not reranker_model: 
        return {"final_docs": docs}

    query = state["query"]
    
    # Prepare pairs for the Cross-Encoder [[Query, Doc_Text], ...]
    pairs = [[query, d.page_content] for d in docs]
    
    logger.info(f"Reranking {len(docs)} documents.")
    # Get scores (0 to 1)
    scores = reranker_model.predict(pairs)
    
    ranked_docs = []
    for i, doc in enumerate(docs):
        # 1. Semantic Score from Cross-Encoder
        semantic_score = scores[i]
        
        # 2. Symbolic Score (Boost if it came from Knowledge Graph)
        symbolic_score = 0.0
        if state.get("kg_movies") and doc.metadata.get("title") in state["kg_movies"]:
            symbolic_score = 0.3  # Boost KG matches
            
        final_score = semantic_score + symbolic_score
        
        # Store score in metadata for debugging
        doc.metadata["rerank_score"] = float(final_score)
        ranked_docs.append((final_score, doc))
        
    # Sort descending
    ranked_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Return top 5
    top_docs = [doc for _, doc in ranked_docs[:5]]
    
    logger.info(f"Reranked top 5 docs.")
    return {"final_docs": top_docs}
