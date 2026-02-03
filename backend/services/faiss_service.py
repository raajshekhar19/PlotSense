"""
FAISS Service for PlotSense backend.
Handles FAISS vector store and dataset operations.
"""
from typing import List, Optional
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, DATASET_PATH
from logger import get_logger

logger = get_logger(__name__)


class FAISSService:
    """Service for managing FAISS vector store and movie dataset."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            FAISSService._initialized = True
    
    def _initialize(self):
        """Initialize FAISS and load dataset."""
        logger.info("Initializing FAISS service...")
        
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )
            logger.info(f"Embeddings model loaded: {EMBEDDING_MODEL}")
            
            # Load FAISS index
            self.movie_db = FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS index loaded from: {FAISS_INDEX_PATH}")
            
            # Load dataset for movie existence checks
            self.df_cleaned = pd.read_csv(DATASET_PATH)
            logger.info(f"Dataset loaded: {DATASET_PATH} ({len(self.df_cleaned)} records)")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS service: {e}")
            raise
    
    def check_movie_exists(self, title: str) -> bool:
        """
        Check if a movie exists in the dataset.
        
        Args:
            title: Movie title to check
            
        Returns:
            True if movie exists, False otherwise
        """
        exists = self.df_cleaned['Title'].str.lower().str.contains(
            title.lower(), na=False
        ).any()
        logger.info(f"Movie '{title}' exists in DB: {exists}")
        return exists
    
    def get_plot_from_dataset(self, title: str) -> Optional[str]:
        """
        Get plot for a movie from the dataset.
        
        Args:
            title: Movie title
            
        Returns:
            Plot string if found, None otherwise
        """
        mask = self.df_cleaned['Title'].str.lower().str.contains(
            title.lower(), na=False
        )
        if mask.any():
            plot = self.df_cleaned.loc[mask, 'Plot'].iloc[0]
            logger.info(f"Retrieved plot from dataset for: {title}")
            return plot
        logger.warning(f"Movie not found in dataset: {title}")
        return None
    
    def similarity_search(self, query: str, k: int = 25) -> List[Document]:
        """
        Perform similarity search on the FAISS index.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        logger.debug(f"Performing similarity search for: {query[:50]}...")
        results = self.movie_db.similarity_search(query, k=k)
        logger.info(f"Similarity search returned {len(results)} results")
        return results
    
    def is_healthy(self) -> bool:
        """Check if FAISS service is healthy."""
        try:
            return self.movie_db is not None and self.df_cleaned is not None
        except Exception as e:
            logger.error(f"FAISS health check failed: {e}")
            return False


# Singleton instance
faiss_service = FAISSService()
