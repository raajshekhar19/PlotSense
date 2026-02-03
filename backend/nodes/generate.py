"""
Answer Generation Nodes for PlotSense backend.
Generates final responses using LLM.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models import MovieState
from services.llm_service import llm_service
from logger import get_logger

logger = get_logger(__name__)


def generate_answer(state: MovieState) -> dict:
    """
    Generate the final answer using retrieved documents.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with final answer
    """
    logger.info("--- Node: Generate Answer ---")
    
    # Check if we found any relevant movies
    docs = state.get("final_docs")
    
    if not docs or len(docs) == 0:
        logger.warning("No docs available for generation.")
        return {
            "final_answer": "I found some movies in my database, but none of them seem to match your description well enough. Could you provide more details about the plot?"
        }
    
    # Prepare context from documents
    context = "\n\n".join(
        d.metadata.get("title", "Unknown") + ": " + d.page_content[:500]
        for d in docs
    )
    
    logger.debug(f"Context passed to LLM (first 200 chars):\n{context[:200]}...")
    
    # Grounded prompt
    prompt = f"""
    You are an authentic, adaptive AI collaborator with a touch of wit. 
    Recommend movies based STRICTLY on the context below. 
    If the context is not helpful, admit you don't know.

    Context:
    {context}

    User Query:
    {state['query']}
    
    Instructions:
    - Briefly explain why these movies fit the user's specific query.
    - Maintain a helpful, peer-like tone.
    """
    
    logger.info("Generating answer...")
    
    answer = llm_service.invoke_gemini(prompt)
    
    logger.info(f"Final answer generated (len: {len(answer)})")
    
    return {"final_answer": answer}


def handle_invalid(state: MovieState) -> dict:
    """
    Handle invalid or gibberish queries.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with error message
    """
    logger.info("--- Node: Handle Invalid/Gibberish ---")
    
    message = (
        "I'm sorry, I didn't quite understand that. "
        "Could you please describe a movie plot, provide a title, "
        "or ask a specific question about actors or genres?"
    )
    
    logger.info(f"Invalid query handled: {state['query'][:50]}...")
    
    return {"final_answer": message}
