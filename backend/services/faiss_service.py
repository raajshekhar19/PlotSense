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
        
        self.movie_db = None
        self.df_cleaned = None
        self.embeddings = None
        
        try:
            # ── Startup validation: verify index files exist before loading ──
            from pathlib import Path
            faiss_dir = Path(FAISS_INDEX_PATH)
            required_files = {
                "index.faiss": faiss_dir / "index.faiss",
                "index.pkl":   faiss_dir / "index.pkl",
            }
            missing = [
                f"  • {name}: expected at {path}"
                for name, path in required_files.items()
                if not path.exists()
            ]
            if missing:
                msg = (
                    f"FAISS index files not found in '{FAISS_INDEX_PATH}'.\n"
                    f"Missing files:\n" + "\n".join(missing) + "\n"
                    f"If running in Docker, make sure the index was copied into the image.\n"
                    f"Run: cp -r artifacts/movie_faiss_v3/ backend/faiss_index/ before 'docker build'."
                )
                logger.error(msg)
                raise FileNotFoundError(msg)
            
            dataset_path = Path(DATASET_PATH)
            if not dataset_path.exists():
                msg = (
                    f"Dataset file not found: '{DATASET_PATH}'.\n"
                    f"If running in Docker, make sure dataset.csv was copied into the image.\n"
                    f"Run: cp dataset.csv backend/dataset.csv before 'docker build'."
                )
                logger.error(msg)
                raise FileNotFoundError(msg)
            
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
            
        except FileNotFoundError:
            # Re-raise file-not-found errors with the clear message above
            raise
        except Exception as e:
            logger.warning(f"FAISS service initialization failed (non-fatal): {e}")
            logger.warning("FAISS search will return empty results until index is available.")
    
    def check_movie_exists(self, title: str) -> bool:
        """
        Check if a movie exists in the dataset.
        
        Args:
            title: Movie title to check
            
        Returns:
            True if movie exists, False otherwise
        """
        if self.df_cleaned is None:
            logger.warning("Dataset not loaded, cannot check movie existence")
            return False
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
        if self.df_cleaned is None:
            logger.warning("Dataset not loaded, cannot get plot")
            return None
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
        if self.movie_db is None:
            logger.warning("FAISS index not loaded, returning empty results")
            return []
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

