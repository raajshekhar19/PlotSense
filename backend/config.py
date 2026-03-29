"""
Configuration module for PlotSense backend.
Loads environment variables and provides centralized configuration.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Load environment variables from .env file (explicit path to project root .env)
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Set environment variables for LangChain
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# ─────────────────────────────────────────
# LangSmith Tracing Configuration
# ─────────────────────────────────────────
LANGSMITH_TRACING   = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT  = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY   = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT   = os.getenv("LANGSMITH_PROJECT", "plotsense")

# Propagate to environment so LangChain SDK picks them up automatically
os.environ["LANGCHAIN_TRACING_V2"]   = LANGSMITH_TRACING
os.environ["LANGCHAIN_ENDPOINT"]      = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_API_KEY"]       = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"]       = LANGSMITH_PROJECT
os.environ["LANGSMITH_TRACING"]       = LANGSMITH_TRACING
os.environ["LANGSMITH_ENDPOINT"]      = LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"]       = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"]       = LANGSMITH_PROJECT

# Model Configuration
GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# FAISS Configuration
FAISS_INDEX_PATH = str(ARTIFACTS_DIR / "movie_faiss_v3")
DATASET_PATH = str(BASE_DIR / "dataset.csv")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = BACKEND_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
