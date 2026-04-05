"""
Knowledge Graph Nodes for PlotSense backend.
Matches hybrid_search_verbose.ipynb exactly.
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from langchain_core.documents import Document
from models import MovieState
from services.neo4j_service import neo4j_service
from services.tavily_service import tavily_service
from services.faiss_service import faiss_service
from services.llm_service import llm_service
from nodes.extract import extract_kg_entities
from config import DATASET_PATH
from logger import get_logger

logger = get_logger(__name__)

# Load dataset using config path
try:
    df1_cleaned = pd.read_csv(DATASET_PATH)
    logger.info(f"kg.py: Dataset loaded from {DATASET_PATH} ({len(df1_cleaned)} records)")
except FileNotFoundError:
    logger.error(f"Could not find dataset at {DATASET_PATH}")
    df1_cleaned = pd.DataFrame(columns=['Title', 'Plot'])

def check_entity_exists(state: MovieState) -> dict:
    """
    Check if the extracted actor/director actually exists in the Neo4j graph.
    If not, we can short-circuit before generating expensive Cypher queries.
    """
    logger.info("--- Node: Check Entity Exists ---")
    entities = state.get("kg_entities", {})
    actor = entities.get('actor')
    director = entities.get('director')
    
    # If no actor/director, we assume it's a genre query, so bypass check
    if not actor and not director:
        return {"entity_not_found": False}
        
    try:
        # Build query to check if name contains the exact keyword
        clauses = []
        if actor: clauses.append(f"toLower(n.name) CONTAINS '{actor.lower()}'")
        if director: clauses.append(f"toLower(n.name) CONTAINS '{director.lower()}'")
        where_clause = " OR ".join(clauses)
        
        cypher = f"MATCH (n) WHERE ('Actor' IN labels(n) OR 'Director' IN labels(n)) AND ({where_clause}) RETURN n LIMIT 1"
        res = neo4j_service.run_cypher(cypher)
        
        if not res:
            logger.warning(f"Entity not found in graph. actor={actor}, director={director}")
            return {"entity_not_found": True}
        
        return {"entity_not_found": False}
    except Exception as e:
        logger.error(f"Entity exists check failed: {e}")
        # Safely fall through if Neo4j is down
        return {"entity_not_found": False}

def handle_missing_entity(state: MovieState) -> dict:
    """
    Handler for when an entity is missing from the graph or Cypher returns 0 results for a person.
    Searches Tavily, skips FAISS, and surfaces the failure immediately if web also fails.
    """
    logger.info("--- Node: Handle Missing Entity ---")
    entities = state.get("kg_entities", {})
    actor = entities.get('actor')
    director = entities.get('director')
    
    person = actor or director or "the requested person"
    query = f"{person} filmography movies directed or starring list"
    
    logger.info(f"Searching web for missing entity: {query}")
    try:
        raw = tavily_service.search_tool.invoke({"query": query})
        results = raw.get("results", [])
        web_context = ""
        for r in results[:2]:
            if r.get("content"):
                web_context += f"Source: {r.get('url', '')}\n{r['content'][:800]}\n\n"
                
        if len(web_context) > 100:
            logger.info("Found missing entity data on the web.")
            return {"web_context": web_context, "search_status": "partial"}
    except Exception as e:
        logger.error(f"Missing entity web search error: {e}")
        
    # If we get here, web failed too.
    logger.warning("Missing entity entirely. Aborting search.")
    return {
        "final_answer": f"We couldn't find {person} in our database or on the web. Try searching by plot description instead.",
        "search_status": "not_found"
    }

def cypher_agent(state: MovieState) -> dict:
    """
    Generate and execute Cypher queries against Neo4j KG.
    Uses run_cypher() for safe execution when Neo4j is unavailable.
    """
    logger.info("--- Node: Cypher Agent ---")
    
    entities = state.get("kg_entities", {})
    logger.info(f"Entities: {entities}")
    
    kg_schema = f"""You are a Neo4j Cypher expert for a movie knowledge graph.

SCHEMA:
  (m:Movie)     props: title (string), year (integer), plot (string)
  (a:Actor)     props: name (string)
  (d:Director)  props: name (string)
  (g:Genre)     props: name (string)  ← compound strings like "Action-Thriller", "Crime / Thriller"
                                         ALWAYS use CONTAINS, never =
  Relationships (outward from Movie):
  (m)-[:FEATURES]->(a:Actor)
  (m)-[:DIRECTED_BY]->(d:Director)
  (m)-[:BELONGS_TO]->(g:Genre)

EXTRACTED ENTITIES:
  actor    = {entities.get('actor')}
  director = {entities.get('director')}
  genre    = {entities.get('genre')}
  year_min = {entities.get('year_min')}
  year_max = {entities.get('year_max')}
  keywords = {entities.get('keywords')}

RULES:
  1. Only MATCH a relationship if its entity is not null above.
  2. ALWAYS use toLower() + CONTAINS for ALL string comparisons. Never use =.
  3. Apply year_min/year_max on m.year if not null.
  4. Always end with: RETURN DISTINCT m.title AS title, m.year AS year ORDER BY m.year DESC LIMIT 15
  5. Return ONLY raw Cypher. No markdown, no backticks, no explanation.

EXAMPLE for actor="amitabh", genre="thriller":
  MATCH (m:Movie)-[:FEATURES]->(a:Actor)
  MATCH (m)-[:BELONGS_TO]->(g:Genre)
  WHERE toLower(a.name) CONTAINS 'amitabh'
  AND toLower(g.name) CONTAINS 'thriller'
  RETURN DISTINCT m.title AS title, m.year AS year
  ORDER BY m.year DESC LIMIT 15

Write the Cypher now:
"""

    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        raw = llm_service.gemini_model.invoke(kg_schema).content.strip()
        cypher = re.sub(r"```cypher|```", "", raw).strip()
        logger.info(f"Attempt {attempt+1} Cypher:\n{cypher}")
        
        try:
            # Use run_cypher() instead of graph.query() directly
            # This safely returns [] when Neo4j is unavailable
            results = neo4j_service.run_cypher(cypher)
            titles = [r["title"] for r in results]
            logger.info(f"KG Found {len(titles)} titles: {titles}")
            
            # Sparse check
            sparse_threshold = 3
            is_sparse = len(titles) < sparse_threshold
            if is_sparse:
                logger.warning(f"Sparse result ({len(titles)} < {sparse_threshold}) — will fallback to web")
                
            return {
                "kg_movies": titles,
                "kg_sparse": is_sparse
            }
        except Exception as e:
            last_error = str(e)
            logger.error(f"Cypher Error: {last_error}")
            kg_schema += f"\nYour previous attempt failed:\nError: {last_error}\nBroken Cypher: {cypher}\nFix it and write only the corrected Cypher:"
            
    # All retries failed — treat as sparse
    return {"kg_movies": [], "kg_sparse": True}

def kg_web_fallback(state: MovieState) -> dict:
    """Web fallback when KG results are sparse — matches notebook."""
    logger.info("--- Node: KG Web Fallback ---")
    
    entities = state.get("kg_entities", {})
    parts = []
    if entities.get('actor'): parts.append(entities['actor'])
    if entities.get('director'): parts.append(f"directed by {entities['director']}")
    if entities.get('genre'): parts.append(entities['genre'])
    if entities.get('year_min') and entities.get('year_max'):
        parts.append(f"from {entities['year_min']} to {entities['year_max']}")
    elif entities.get('year_min'):
        parts.append(f"after {entities['year_min']}")
    if entities.get('keywords'): parts.append(" ".join(entities['keywords']))
        
    structured_query = " ".join(parts) + " best movies list"
    logger.info(f"Structured search query: {structured_query}")
    
    search_queries = [structured_query, f"{structured_query} site:imdb.com"]
    web_context = ""
    
    for q in search_queries:
        try:
            raw = tavily_service.search_tool.invoke({"query": q})
            results = raw.get("results", [])
            for r in results[:2]:
                if r.get("content"):
                    web_context += f"Source: {r.get('url', '')}\n{r['content'][:800]}\n\n"
            if len(web_context) > 1500: break
        except Exception as e:
            logger.error(f"Search error for '{q}': {e}")
            continue

    if not web_context:
        logger.warning("No web results found")
        return {"web_context": None}

    logger.info(f"Web context retrieved ({len(web_context)} chars)")
    return {
        "web_context": web_context,
        "base_plot": web_context  # keep base_plot in sync for similarity_search fallback
    }

def kg_results_to_docs(state: MovieState) -> dict:
    """Convert KG movie titles to Document objects — matches notebook."""
    logger.info("--- Node: KG Results to Docs ---")
    
    titles = state.get("kg_movies", [])
    docs = []
    
    for title in titles:
        # 1. Try exact match in df1_cleaned first
        mask = df1_cleaned['Title'].str.lower() == title.lower()
        if not mask.any():
            # 2. Fallback: contains match
            mask = df1_cleaned['Title'].str.lower().str.contains(title.lower(), na=False, regex=False)
            
        if mask.any():
            row = df1_cleaned[mask].iloc[0]
            plot = row.get('Plot') or row.get('clean_plot') or ""
            docs.append(Document(page_content=str(plot), metadata={"title": row['Title']}))
        else:
            # 3. Last resort: FAISS (only if title not in CSV)
            faiss_matches = faiss_service.similarity_search(title, k=1)
            if faiss_matches: docs.append(faiss_matches[0])
            
    logger.info(f"Fetched {len(docs)} docs for {len(titles)} KG titles")
    return {"final_docs": docs}
