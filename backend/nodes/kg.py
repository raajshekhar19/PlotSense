"""
Knowledge Graph Nodes for PlotSense backend.
Handles KG queries and result processing.
"""
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])

from models import MovieState
from services.llm_service import llm_service
from services.neo4j_service import neo4j_service
from services.faiss_service import faiss_service
from logger import get_logger

logger = get_logger(__name__)


def kg_agent(state: MovieState) -> dict:
    """
    Advanced Knowledge Graph agent for complex queries.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with KG movie titles
    """
    logger.info("--- Node: Advanced KG Agent ---")
    
    logger.info(f"Extracting entities from query: {state['query']}")
    
    # Extract entities using Gemini structured output
    entities = llm_service.extract_filters(state['query'])
    
    logger.info(f"Extracted Entities:")
    logger.info(f"  Actor: {entities.actor}")
    logger.info(f"  Director: {entities.director}")
    logger.info(f"  Genre: {entities.genre}")
    logger.info(f"  Keywords: {entities.keywords}")
    
    # Dynamic Cypher to handle any combination of actor, director, genre, or plot
    cypher = """
    MATCH (m:Movie)
    OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Person)
    OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
    OPTIONAL MATCH (m)-[:IN_GENRE]->(g:Genre)
    WHERE
        ($actor IS NULL OR toLower(a.name) CONTAINS toLower($actor)) AND
        ($director IS NULL OR toLower(d.name) CONTAINS toLower($director)) AND
        ($genre IS NULL OR toLower(g.name) CONTAINS toLower($genre)) AND
        ($keywords IS NULL OR ANY(k IN $keywords WHERE toLower(m.plot) CONTAINS toLower(k)))
    RETURN DISTINCT m.title AS title
    LIMIT 15
    """
    
    params = {
        "actor": entities.actor,
        "director": entities.director,
        "genre": entities.genre,
        "keywords": entities.keywords if entities.keywords else None
    }
    
    logger.info(f"Generated Cypher Params: {params}")
    
    data = neo4j_service.run_cypher(cypher, params)
    titles = [d["title"] for d in data]
    
    logger.info(f"KG Found Titles: {titles}")
    
    return {"kg_movies": titles}


def kg_results_to_docs(state: MovieState) -> dict:
    """
    Convert KG movie titles to document objects.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with final documents
    """
    logger.info("--- Node: KG Results to Docs ---")
    
    titles = state.get("kg_movies", [])
    docs = []
    
    for title in titles:
        matches = faiss_service.similarity_search(title, k=1)
        if matches:
            docs.append(matches[0])
    
    logger.info(f"Converted {len(docs)} KG titles to documents.")
    
    return {"final_docs": docs}
