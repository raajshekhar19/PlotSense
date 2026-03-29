"""
Query Classification Node for PlotSense backend.
Classifies user intent using robust JSON fallback — matches notebook.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MovieState
from services.llm_service import llm_service
from logger import get_logger

logger = get_logger(__name__)


import re
import json

def classify_query(state: MovieState) -> dict:
    """
    Classify the user's movie query intent with robust fallback.
    """
    logger.info("--- Node: Classify Query ---")
    
    prompt = f"""You are a Movie Intent Classifier. Analyze the query and return ONLY a JSON object.

INTENTS:
- "plot"         : user describes a story/scene (e.g. "movie where a man grows potatoes on Mars")
- "movie_name"   : user names a specific movie and wants similar ones (e.g. "movies like Inception")
- "query_search" : user filters by actor / director / genre / year (e.g. "Tom Hanks action movies")
- "invalid"      : gibberish or completely off-topic

Return ONLY this JSON, nothing else:
{{"intent": "plot|movie_name|query_search|invalid", "confidence_score": 0.0}}

Query: "{state['query']}"
"""
    
    raw = llm_service.gemini_model.invoke(prompt).content.strip()
    
    # Strip accidental markdown
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        intent = parsed.get("intent", "invalid")
        score  = parsed.get("confidence_score", 0.0)
    except json.JSONDecodeError:
        # Fallback: scan raw text for intent keyword
        intent = "invalid"
        for candidate in ["plot", "movie_name", "query_search"]:
            if candidate in raw.lower():
                intent = candidate
                break
        score = 0.5
        
    logger.info(f"Identified Intent: {intent} (Confidence: {score})")
    return {"intent": intent}
