"""
Query Classification Node for PlotSense backend.
Classifies user intent using Gemini structured output.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models import MovieState
from services.llm_service import llm_service
from logger import get_logger

logger = get_logger(__name__)


def classify_query(state: MovieState) -> dict:
    """
    Classify the user's movie query intent.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with classified intent
    """
    logger.info("--- Node: Classify Query (Gemini Structured) ---")
    
    prompt = f"""
    ### ROLE
    You are a highly accurate Movie Intent Classifier. Your goal is to categorize user input so the system can choose the correct search path.

    ### INTENT DEFINITIONS
    1. 'plot': The user is describing specific events, scenes, or storylines (e.g., "movie where a man grows potatoes on Mars").
    2. 'movie_name': The user provides a specific, recognizable movie title. They want recommendations SIMILAR to this movie or based on its plot.
       - INCLUDES: "Movies like Inception", "Something similar to Titanic", "Recommendations based on Matrix".
    3. 'query_search': The user is asking for a list based on ACTORS, DIRECTORS, GENRES, or YEARS.
       - INCLUDES: "Movies with Tom Hanks", "Best 90s thrillers", "Sci-fi movies directed by Nolan".
    4. 'invalid': Gibberish, random characters, or completely unrelated to movies.

    ### EXAMPLES
    - "A movie where a man is trapped on Mars" -> plot
    - "Suggest me movie based on movie Titanic" -> movie_name
    - "Movies like Metro in Dino" -> movie_name
    - "Similar to Interstellar" -> movie_name
    - "Action movies starring Keanu Reeves" -> query_search
    - "Best horror from 2023" -> query_search

    ### INSTRUCTION
    Analyze the query below. 
    - If the user names a SPECIFIC MOVIE to get similar recommendations (e.g., "Like [Movie Name]"), classify as 'movie_name'.
    - Only classify as 'query_search' if they are filtering by Actor, Director, Genre, or Year.
    
    User Query: "{state['query']}"
    """
    
    logger.debug(f"Classification prompt length: {len(prompt)}")
    
    response = llm_service.classify_intent(prompt)
    
    logger.info(f"Identified Intent: {response.intent} (Confidence: {response.confidence_score})")
    
    return {"intent": response.intent}
