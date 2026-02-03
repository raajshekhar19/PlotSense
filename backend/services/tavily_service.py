"""
Tavily Service for PlotSense backend.
Handles web search for movie plot retrieval.
"""
import re
from typing import Optional
from langchain_tavily import TavilySearch

from logger import get_logger

logger = get_logger(__name__)


class TavilyService:
    """Service for web search operations using Tavily."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            TavilyService._initialized = True
    
    def _initialize(self):
        """Initialize Tavily search client."""
        logger.info("Initializing Tavily service...")
        
        try:
            self.search_tool = TavilySearch(
                max_results=1,
                topic="general",
                include_raw_content=True,
                include_answer=True,
                search_depth="advanced",
                time_range=None,
                include_images=True,
            )
            logger.info("Tavily search client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily service: {e}")
            raise
    
    @staticmethod
    def clean_text(text) -> str:
        """
        Clean extracted text from web search.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if isinstance(text, list):
            text = " ".join(map(str, text))
        if not isinstance(text, str):
            text = str(text)
        
        text = text.replace("\\n", " ").replace("\n", " ").replace("\t", " ")
        text = re.sub(r"\[\d+\]", "", text)  # remove citations [1]
        text = re.sub(r"\[|\]", "", text)    # remove stray brackets
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def search_movie_plot(self, movie_name: str) -> Optional[str]:
        """
        Search for a movie's plot on the web.
        
        Args:
            movie_name: Name of the movie to search
            
        Returns:
            Plot description if found, None otherwise
        """
        # Normalize movie name
        if isinstance(movie_name, dict):
            movie_name = movie_name.get("text", "")
        elif not isinstance(movie_name, str):
            movie_name = str(movie_name)
        
        movie_name = movie_name.strip()
        
        if not movie_name:
            logger.warning("No movie name provided for web search")
            return None
        
        try:
            search_query = f"movie {movie_name} plot summary wikipedia"
            logger.info(f"Searching Tavily with query: {search_query}")
            
            raw_response = self.search_tool.invoke({"query": search_query})
            
            logger.debug(f"Tavily response type: {type(raw_response)}")
            if isinstance(raw_response, dict):
                logger.debug(f"Tavily response keys: {raw_response.keys()}")
            
            # Extract content safely
            results = raw_response.get("results", []) if isinstance(raw_response, dict) else []
            
            target_content = ""
            source_url = ""
            
            for r in results:
                if isinstance(r, dict) and r.get("content"):
                    target_content = r["content"]
                    source_url = r.get("url", "Unknown Source")
                    break
            
            if not target_content:
                logger.warning(f"No plot found for {movie_name}")
                return None
            
            logger.info(f"Plot source: {source_url}")
            
            plot = self.clean_text(target_content)
            
            if not plot:
                logger.warning(f"Plot extraction failed after cleaning for {movie_name}")
                return None
            
            logger.info(f"Successfully retrieved plot for {movie_name}")
            logger.debug(f"Extracted Plot:\n{plot[:200]}...")
            
            return plot
            
        except Exception as e:
            logger.error(f"Web Search Error: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """Check if Tavily service is healthy."""
        try:
            return self.search_tool is not None
        except Exception as e:
            logger.error(f"Tavily health check failed: {e}")
            return False


# Singleton instance
tavily_service = TavilyService()
