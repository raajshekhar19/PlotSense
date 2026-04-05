"""
Similarity and Web Search Nodes for PlotSense backend.
Matches hybrid_search_verbose.ipynb exactly.
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MovieState
from services.faiss_service import faiss_service
from services.tavily_service import tavily_service
from services.llm_service import llm_service
from nodes.extract import extract_kg_entities
from logger import get_logger

logger = get_logger(__name__)

def get_web_plot(state: MovieState) -> dict:
    """Get the movie plot from web search."""
    logger.info("--- Node: Get Web Plot ---")

    movie_name = state.get("movie_name", "")
    if isinstance(movie_name, dict):
        movie_name = movie_name.get("text", "")
    elif not isinstance(movie_name, str):
        movie_name = str(movie_name)
    movie_name = movie_name.strip()

    if not movie_name:
        return {"base_plot": "No movie name provided."}

    try:
        search_query = f"movie {movie_name} plot summary wikipedia"
        logger.info(f"Searching Tavily: {search_query}")
        raw_response = tavily_service.search_tool.invoke({"query": search_query})

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
            return {"base_plot": "No plot found on the web."}

        logger.info(f"Plot source: {source_url}")

        def clean_text(text) -> str:
            if isinstance(text, list): text = " ".join(map(str, text))
            if not isinstance(text, str): text = str(text)
            text = text.replace("\\n", " ").replace("\n", " ").replace("\t", " ")
            text = re.sub(r"\[\d+\]", "", text)
            text = re.sub(r"\[|\]", "", text)
            return re.sub(r"\s+", " ", text).strip()

        plot = clean_text(target_content)
        if not plot: return {"base_plot": "Plot extraction failed."}
        
        logger.info(f"Successfully retrieved plot for {movie_name}")
        return {"base_plot": plot}

    except Exception as e:
        logger.error(f"Web Search Error: {e}")
        return {"base_plot": f"Error: {e}"}

def strip_intent_language(state: MovieState) -> dict:
    """
    Remove functional navigation vocabulary (like 'show me movies by', 'directed by') 
    so FAISS only receives pure semantic keywords.
    """
    logger.info("--- Node: Strip Intent Language ---")
    
    query = state["query"]
    prompt = f"""You are a query optimizer for a semantic vector database.
The user asked: "{query}"

Strip away ALL navigational language, conversational filler, and explicit structural words.
Remove phrases like:
- "show me"
- "movies by"
- "directed by"
- "films starring"
- "list of"
- "I want to watch"

Return ONLY the core semantic keywords or plot descriptions remaining.
If stripping leaves absolutely nothing (e.g., query was literally just "movies"), return the original query.

Cleaned query:"""
    
    try:
        response = llm_service.gemini_model.invoke(prompt)
        cleaned = response.content.strip()
        logger.info(f"Stripped query from '{query}' -> '{cleaned}'")
        return {"base_plot": cleaned}
    except Exception as e:
        logger.error(f"Intent strip failed: {e}")
        return {"base_plot": query}

def similarity_search(state: MovieState) -> dict:
    """Perform aggregated similarity search — matches notebook."""
    logger.info("--- Node: Similarity Search (aggregated) ---")
    
    query_text = state.get("base_plot") or state["query"]
    
    # Fetch MORE chunks than needed (k=25) to get chunks from different movies
    raw_docs = faiss_service.similarity_search(query_text, k=25)
    
    # Group chunks by Movie Name (Deduplicate)
    unique_movies = {}
    for doc in raw_docs:
        title = doc.metadata.get("title", "Unknown")
        if title not in unique_movies:
            unique_movies[title] = {
                "doc": doc,
                "score": 0.0,
                "chunks": [doc.page_content]
            }
        else:
            unique_movies[title]["chunks"].append(doc.page_content)
    
    # Limit to top 5 UNIQUE movies
    final_docs = [data["doc"] for title, data in list(unique_movies.items())[:5]]
    logger.info(f"Aggregated to {len(final_docs)} unique movies.")
    
    return {"final_docs": final_docs}

def generate_from_web(state: MovieState) -> dict:    
    """Generate answer from web context when KG results are sparse — matches notebook."""
    logger.info("--- Node: Generate From Web ---")

    web_context = state.get("web_context")
    entities    = extract_kg_entities(state)

    if not web_context:
        return {
            "final_answer": (
                f"I couldn't find enough results for '{state['query']}' "
                "in my database or on the web. Try rephrasing or being more specific."
            )
        }

    # Build constraint string for the LLM prompt
    constraints = []
    if entities.get('actor'): constraints.append(f"starring {entities['actor']}")
    if entities.get('director'): constraints.append(f"directed by {entities['director']}")
    if entities.get('genre'): constraints.append(f"in the {entities['genre']} genre")
    if entities.get('year_min') and entities.get('year_max'):
        constraints.append(f"released between {entities['year_min']} and {entities['year_max']}")
    elif entities.get('year_min'):
        constraints.append(f"released after {entities['year_min']}")
    constraint_str = ", ".join(constraints) if constraints else "matching the user's request"

    prompt = f"""You are a helpful movie recommendation assistant.

The user asked: "{state['query']}"

I searched the web and found this information:
{web_context}

Your job:
- Recommend movies {constraint_str}
- Only mention movies explicitly found in the web content above
- If the web content mentions movies that do NOT match the constraints (wrong year, wrong actor), skip them
- If you genuinely cannot find enough matching movies, say so honestly and suggest the user try a more specific search
- List each movie with title, year, and one sentence on why it fits
- Keep it conversational
"""
    answer = llm_service.gemini_model.invoke(prompt).content
    logger.info(f"Web-based answer generated (len: {len(answer)})")
    return {"final_answer": answer}
