"""
Movie Name Extraction Nodes for PlotSense backend.
Extracts movie names and checks existence in database — matches notebook.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MovieState
from services.llm_service import llm_service
from services.faiss_service import faiss_service
from config import DATASET_PATH
from logger import get_logger

logger = get_logger(__name__)


import re
import json
import pandas as pd

# Load the local dataset into memory once using config path
try:
    df1_cleaned = pd.read_csv(DATASET_PATH)
    logger.info(f"extract.py: Dataset loaded from {DATASET_PATH} ({len(df1_cleaned)} records)")
except FileNotFoundError:
    logger.error(f"Could not find dataset at {DATASET_PATH}")
    df1_cleaned = pd.DataFrame(columns=['Title', 'Plot'])

def extract_movie_name(state: MovieState) -> dict:
    """
    Extract the specific movie name from the user's query.
    """
    logger.info("--- Node: Extract Movie Name ---")
    
    prompt = f"""Extract the movie title from the user's query.

Rules:
- Extract whatever title the user mentions, even if you don't recognize it
- The movie may be regional, recent, or obscure — extract it anyway
- If truly no movie title is present, respond with NONE
- Return ONLY the movie title, nothing else

Examples:
"movies like Dhurandhar"          → Dhurandhar
"something similar to RRR"        → RRR  
"recommend films like KGF"        → KGF
"movies like Metro In Dino"       → Metro In Dino
"I want a thriller"               → NONE

Query: {state['query']}
"""
    
    response = llm_service.gemini_model.invoke(prompt)
    name = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
    name = re.sub(r"```.*?```", "", name, flags=re.DOTALL).strip()
    
    logger.info(f"Extracted Name: {name}")
    
    if "NONE" in name.upper() or not name:
        return {"movie_name": None}
    
    return {"movie_name": name}

def check_movie_exists(state: MovieState) -> dict:
    """
    Check if the extracted movie exists in the pandas dataset.
    """
    logger.info("--- Node: Check Movie Exists ---")
    
    title = state["movie_name"]
    # Case-insensitive substring match
    exists = df1_cleaned['Title'].str.lower().str.contains(title.lower(), na=False).any()
    
    logger.info(f"Movie Exists in DB: {exists}")
    return {"movie_exists": bool(exists)}


def extract_kg_entities(state: MovieState) -> dict:
    """
    Extract structured entities from user query for KG search.
    Called by kg.py and search.py with full state dict.
    """
    prompt = f"""Extract movie search filters from the query. Return ONLY a JSON object.

Keys (use null if not mentioned):
- actor    : most distinctive part of actor name only (e.g. "amitabh" not "Amitabh Bachchan")
- director : most distinctive part of director name only (e.g. "nolan" not "Christopher Nolan")
- genre    : single most relevant genre keyword (e.g. "thriller", "horror", "comedy")
- year_min : minimum year as integer or null
- year_max : maximum year as integer or null  
- keywords : list of plot keywords or null

CRITICAL RULES:
- For actor/director: use ONLY the most unique part of the name (first OR last name, not both)
  "Amitabh Bachchan" → "amitabh"
  "Tom Hanks"        → "hanks"  
  "Shah Rukh Khan"   → "shah rukh"
  "Christopher Nolan"→ "nolan"
- For genre: use a single lowercase keyword that will appear inside compound genres
  "thriller" will match "Action-Thriller", "Crime/Thriller", "Psychological Thriller" etc.
  "horror"   will match "Horror/Thriller", "Horror Drama" etc.
- For decades: "90s" → year_min:1990, year_max:1999 | "2000s" → year_min:2000, year_max:2009
- For "recent/latest/new" → year_min:2015, year_max:null

Examples:
"Amitabh Bachchan thriller"
→ {{"actor":"amitabh","director":null,"genre":"thriller","year_min":null,"year_max":null,"keywords":null}}

"Shah Rukh Khan romantic movies from 2000s"
→ {{"actor":"shah rukh","director":null,"genre":"romance","year_min":2000,"year_max":2009,"keywords":null}}

"Nolan psychological films"
→ {{"actor":null,"director":"nolan","genre":"psychological","year_min":null,"year_max":null,"keywords":null}}

Query: "{state['query']}"
"""
    raw = llm_service.gemini_model.invoke(prompt).content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Entity extraction parse failed, raw: {raw}")
        return {"actor": None, "director": None, "genre": None,
                "year_min": None, "year_max": None, "keywords": None}
