"""
Movie Name Extraction Nodes for PlotSense backend.
Extracts movie names and checks existence in database.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models import MovieState
from services.llm_service import llm_service
from services.faiss_service import faiss_service
from logger import get_logger

logger = get_logger(__name__)


def extract_movie_name(state: MovieState) -> dict:
    """
    Extract the specific movie name from the user's query.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with extracted movie name
    """
    logger.info("--- Node: Extract Movie Name ---")
    
    prompt = f"""
    Extract the specific movie name from the query.
    If no specific movie name is found, respond EXACTLY with "NONE".
    Query:
    {state['query']}
    """
    
    logger.debug(f"Extraction prompt: {prompt.strip()}")
    
    name = llm_service.invoke_gemini(prompt)
    
    logger.info(f"Extracted Name: {name}")
    
    # Logic to return None if no name found
    if "NONE" in name.upper() or not name:
        logger.warning("No movie name could be extracted")
        return {"movie_name": None}
    
    return {"movie_name": name}


def check_movie_exists(state: MovieState) -> dict:
    """
    Check if the extracted movie exists in the dataset.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with movie existence flag
    """
    logger.info("--- Node: Check Movie Exists ---")
    
    title = state["movie_name"]
    exists = faiss_service.check_movie_exists(title)
    
    logger.info(f"Movie Exists in DB: {exists}")
    
    return {"movie_exists": exists}
