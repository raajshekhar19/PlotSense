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
    WHERE
        ($actor IS NULL OR EXISTS { MATCH (m)-[:FEATURES]->(a:Actor) WHERE toLower(a.name) CONTAINS toLower($actor) }) AND
        ($director IS NULL OR EXISTS { MATCH (m)-[:DIRECTED_BY]->(d:Director) WHERE toLower(d.name) CONTAINS toLower($director) }) AND
        ($genre IS NULL OR EXISTS { MATCH (m)-[:BELONGS_TO]->(g:Genre) WHERE toLower(g.name) CONTAINS toLower($genre) }) AND
        ($keywords IS NULL OR size($keywords)=0 OR ANY(k IN $keywords WHERE toLower(m.plot) CONTAINS toLower(k)))
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
    Convert KG movie titles to document objects directly via Neo4j.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dictionary with final documents
    """
    logger.info("--- Node: KG Results to Docs ---")
    
    titles = state.get("kg_movies", [])
    docs = []
    
    if titles:
        # Fetch the plots directly from Neo4j for the matched titles
        cypher = """
        MATCH (m:Movie)
        WHERE m.title IN $titles
        RETURN m.title AS title, m.plot AS plot
        """
        data = neo4j_service.run_cypher(cypher, {"titles": titles})
        
        from langchain_core.documents import Document
        
        for record in data:
            if record.get("plot"):
                docs.append(Document(
                    page_content=record["plot"],
                    metadata={"title": record["title"], "source": "Neo4j"}
                ))
                
    # Fallback to Tavily if Neo4j returns absolutely nothing
    if not docs:
        logger.warning("Neo4j KG returned 0 results. Falling back to Tavily Web Search...")
        try:
            from services.tavily_service import tavily_service
            query_text = state.get("query", "")
            search_query = f"{query_text} movie title"
            raw_response = tavily_service.search_tool.invoke({"query": search_query})
            
            results = raw_response.get("results", []) if isinstance(raw_response, dict) else []
            
            from langchain_core.documents import Document
            for r in results[:5]:
                if isinstance(r, dict) and r.get("content"):
                    docs.append(Document(
                        page_content=r["content"],
                        metadata={"title": r.get("title", "Web Result"), "source": "Tavily Web Search", "url": r.get("url", "")}
                    ))
            
            if docs:
                logger.info(f"Retrieved {len(docs)} fallback documents from Tavily.")
        except Exception as e:
            logger.error(f"Tavily fallback failed: {e}")
    
    logger.info(f"Converted {len(docs)} KG titles to documents using Neo4j (or Web Fallback).")
    
    return {"final_docs": docs}
