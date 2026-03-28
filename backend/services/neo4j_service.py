"""
Neo4j Service for PlotSense backend.
Handles Neo4j graph database connections and queries.
"""
from typing import List, Dict, Any, Optional
from langchain_neo4j import Neo4jGraph

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from logger import get_logger

logger = get_logger(__name__)


class Neo4jService:
    """Service for managing Neo4j database interactions."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            Neo4jService._initialized = True
    
    def _initialize(self):
        """Initialize Neo4j connection (non-fatal on failure)."""
        logger.info("Initializing Neo4j connection...")
        self.graph = None
        self._connected = False
        
        try:
            self.graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                refresh_schema=False
            )
            self._connected = True
            logger.info(f"Neo4j connected successfully: {NEO4J_URI}")
        except Exception as e:
            logger.error(
                f"Neo4j connection failed — service marked unhealthy. "
                f"Wake up your Aura instance at console.neo4j.io\nError: {e}"
            )
            # Do NOT re-raise: let the app start without Neo4j
    
    def run_cypher(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against Neo4j.
        
        Args:
            query: The Cypher query string
            params: Optional query parameters
            
        Returns:
            List of result records as dictionaries
        """
        logger.info("--- Executing Cypher ---")
        logger.debug(f"Query:\n{query}")
        logger.debug(f"Params: {params}")
        
        if not self._connected or self.graph is None:
            logger.warning("Neo4j is not connected — skipping Cypher query, returning empty results")
            return []
        
        try:
            results = self.graph.query(query, params or {})
            logger.info(f"Cypher Result Count: {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Cypher Execution Error: {e}")
            return []
    
    def is_healthy(self) -> bool:
        """Check if Neo4j connection is healthy."""
        if not self._connected or self.graph is None:
            return False
        try:
            self.graph.query("RETURN 1 as healthy")
            return True
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False


# Singleton instance
neo4j_service = Neo4jService()
