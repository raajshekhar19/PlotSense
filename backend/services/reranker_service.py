"""
Reranker Service for PlotSense backend.
Handles cross-encoder reranking operations.
"""
from typing import List, Tuple, Any
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL
from logger import get_logger

logger = get_logger(__name__)


class RerankerService:
    """Service for cross-encoder reranking operations."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            RerankerService._initialized = True
    
    def _initialize(self):
        """Initialize the cross-encoder model."""
        logger.info("Initializing Reranker service...")
        
        try:
            self.model = CrossEncoder(RERANKER_MODEL)
            logger.info(f"CrossEncoder model loaded: {RERANKER_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Reranker service: {e}")
            raise
    
    def rerank(self, query: str, documents: List[Any], kg_movies: List[str] = None) -> List[Any]:
        """
        Rerank documents using cross-encoder and KG boost.
        
        Args:
            query: User's search query
            documents: List of documents to rerank
            kg_movies: List of movie titles from KG (for boosting)
            
        Returns:
            Top 5 reranked documents
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        logger.info(f"--- Reranking {len(documents)} documents ---")
        
        # Prepare pairs for cross-encoder
        pairs = [[query, d.page_content] for d in documents]
        
        # Get semantic scores
        scores = self.model.predict(pairs)
        
        ranked_docs = []
        for i, doc in enumerate(documents):
            # Semantic score from cross-encoder
            semantic_score = scores[i]
            
            # Symbolic score (boost KG matches)
            symbolic_score = 0.0
            if kg_movies and doc.metadata.get("title") in kg_movies:
                symbolic_score = 0.3
                logger.debug(f"KG boost applied to: {doc.metadata.get('title')}")
            
            final_score = semantic_score + symbolic_score
            
            # Store score in metadata
            doc.metadata["rerank_score"] = float(final_score)
            ranked_docs.append((final_score, doc))
        
        # Sort descending
        ranked_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top 5
        top_docs = [doc for _, doc in ranked_docs[:5]]
        logger.info(f"Reranking complete. Top score: {ranked_docs[0][0]:.4f}")
        
        return top_docs
    
    def is_healthy(self) -> bool:
        """Check if Reranker service is healthy."""
        try:
            return self.model is not None
        except Exception as e:
            logger.error(f"Reranker health check failed: {e}")
            return False


# Singleton instance
reranker_service = RerankerService()
