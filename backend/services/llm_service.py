"""
LLM Service for PlotSense backend.
Handles Groq model initialization and invocation.
Matches hybrid_search_verbose.ipynb exactly.
"""
from langchain_groq import ChatGroq
from config import GROQ_MODEL
from logger import get_logger
from models import IntentClassification, MovieFilters

logger = get_logger(__name__)


class LLMService:
    """Service for managing LLM interactions."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            LLMService._initialized = True
    
    def _initialize(self):
        """Initialize LLM model — single ChatGroq (matches notebook)."""
        logger.info("Initializing LLM models...")
        
        try:
            # ChatGroq model — matches notebook exactly
            self.gemini_model = ChatGroq(
                model=GROQ_MODEL,
                temperature=1.0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            logger.info(f"Groq model initialized: {GROQ_MODEL}")
            
            # Structured output models
            self.structured_classify = self.gemini_model.with_structured_output(IntentClassification)
            self.structured_extractor = self.gemini_model.with_structured_output(MovieFilters)
            logger.info("Structured output models configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM models: {e}")
            raise
    
    def classify_intent(self, prompt: str) -> IntentClassification:
        """
        Classify user query intent using structured output.
        
        Args:
            prompt: The classification prompt
            
        Returns:
            IntentClassification with intent and confidence
        """
        logger.debug(f"Classifying intent for prompt length: {len(prompt)}")
        result = self.structured_classify.invoke(prompt)
        logger.info(f"Intent classified: {result.intent} (confidence: {result.confidence_score})")
        return result
    
    def extract_filters(self, query: str) -> MovieFilters:
        """
        Extract movie filters from query using structured output.
        
        Args:
            query: User's movie query
            
        Returns:
            MovieFilters with extracted entities
        """
        logger.debug(f"Extracting filters from query: {query}")
        result = self.structured_extractor.invoke(query)
        logger.info(f"Extracted filters - Actor: {result.actor}, Director: {result.director}, Genre: {result.genre}")
        return result
    
    def invoke_gemini(self, prompt: str) -> str:
        """
        Invoke Groq model for general text generation.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Generated text response
        """
        logger.debug("Invoking Groq model")
        response = self.gemini_model.invoke(prompt)
        
        # Handle response content
        if isinstance(response.content, list):
            parts = []
            for item in response.content:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            return " ".join(parts).strip()
        return str(response.content).strip()
    
    def is_healthy(self) -> bool:
        """Check if LLM service is healthy."""
        try:
            return self.gemini_model is not None
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False


# Singleton instance
llm_service = LLMService()
