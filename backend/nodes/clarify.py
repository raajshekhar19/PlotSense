"""
Clarification Node for PlotSense backend.
Matches hybrid_search_verbose.ipynb exactly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MovieState
from services.llm_service import llm_service
from logger import get_logger

logger = get_logger(__name__)

def clarification_node(state: MovieState) -> dict:
    """
    Triggered when: movie name extracted but not in DB and web found nothing.
    Asks user to confirm spelling or provide more context.
    """
    logger.info("--- Node: Clarification ---")

    movie_name = state.get("movie_name", "")
    base_plot  = state.get("base_plot", "")

    # If web search returned something, no clarification needed
    if base_plot and len(base_plot) > 100:
        return {"needs_clarification": False}

    # Build a smart clarification question using LLM
    prompt = f"""The user asked for movies similar to "{movie_name}", 
but I couldn't find this movie in my database or on the web.

Generate a short, friendly clarification message that:
1. Acknowledges the movie wasn't found
2. Asks if the spelling is correct OR if they can describe the movie's plot
3. Suggests they might mean a similar well-known title if applicable
4. Keeps it under 3 sentences

Return ONLY the clarification message, nothing else.
"""
    
    question = llm_service.gemini_model.invoke(prompt).content.strip()
    logger.info(f"Clarification question: {question}")

    return {
        "needs_clarification": True,
        "clarification_question": question,
        "final_answer": question   # surfaces to user immediately
    }
