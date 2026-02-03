"""
Plot Retrieval Nodes for PlotSense backend.
Gets movie plots from dataset or web.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models import MovieState
from services.faiss_service import faiss_service
from services.tavily_service import tavily_service
from logger import get_logger

logger = get_logger(__name__)


def get_dataset_plot(state: MovieState) -> dict:
    """
    Get the movie plot from the local dataset.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with base plot
    """
    logger.info("--- Node: Get Dataset Plot ---")
    
    title = state["movie_name"]
    plot = faiss_service.get_plot_from_dataset(title)
    
    if plot:
        logger.info("Retrieved plot from Dataset.")
        logger.debug(f"Plot preview: {plot[:200]}...")
    else:
        logger.warning("Movie not found in Dataset.")
    
    return {"base_plot": plot}


def get_web_plot(state: MovieState) -> dict:
    """
    Get the movie plot from web search.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with base plot
    """
    logger.info("--- Node: Get Web Plot ---")
    
    movie_name = state.get("movie_name", "")
    
    if not movie_name:
        logger.warning("No movie name provided for web search")
        return {"base_plot": "No movie name provided."}
    
    plot = tavily_service.search_movie_plot(movie_name)
    
    if plot:
        logger.info(f"Successfully retrieved plot for {movie_name}")
        logger.debug(f"Extracted Plot:\n{plot[:200]}...")
        return {"base_plot": plot}
    else:
        logger.warning(f"Failed to retrieve plot for {movie_name}")
        return {"base_plot": "No plot found on the web."}
