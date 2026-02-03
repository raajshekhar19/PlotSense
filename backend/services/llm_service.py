"""
LLM Service for PlotSense backend.
Handles Gemini and Ollama model initialization and invocation.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from config import GEMINI_MODEL, OLLAMA_MODEL
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
        """Initialize LLM models."""
        logger.info("Initializing LLM models...")
        
        try:
            # Gemini model for classification and extraction
            self.gemini_model = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                temperature=1.0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            logger.info(f"Gemini model initialized: {GEMINI_MODEL}")
            
            # Structured output models
            self.structured_classify = self.gemini_model.with_structured_output(IntentClassification)
            self.structured_extractor = self.gemini_model.with_structured_output(MovieFilters)
            logger.info("Structured output models configured")
            
            # Ollama model for answer generation
            self.ollama_model = ChatOllama(model=OLLAMA_MODEL)
            logger.info(f"Ollama model initialized: {OLLAMA_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM models: {e}")
            raise
    
    def classify_intent(self, prompt: str) -> IntentClassification:
        """
        Classify user query intent using Gemini.
        
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
        Extract movie filters from query using Gemini.
        
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
        Invoke Gemini model for general text generation.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Generated text response
        """
        logger.debug("Invoking Gemini model")
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
    
    def invoke_ollama(self, prompt: str) -> str:
        """
        Invoke Ollama model for answer generation.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Generated text response
        """
        logger.debug("Invoking Ollama model")
        response = self.ollama_model.invoke(prompt)
        return response.content
    
    def is_healthy(self) -> bool:
        """Check if LLM service is healthy."""
        try:
            # Simple health check
            return self.gemini_model is not None and self.ollama_model is not None
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False


# Singleton instance
llm_service = LLMService()
