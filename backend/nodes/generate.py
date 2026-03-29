"""
Answer Generation Nodes for PlotSense backend.
Generates final responses using LLM — matches notebook exactly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MovieState
from services.llm_service import llm_service
from logger import get_logger

logger = get_logger(__name__)


def generate_answer(state: MovieState) -> dict:
    """
    Generate the final answer using retrieved documents and KG-confirmed titles.
    Matches notebook exactly.
    """
    logger.info("--- Node: Generate Answer ---")

    docs      = state.get("final_docs", [])
    kg_movies = state.get("kg_movies", [])

    if not docs:
        return {
            "final_answer": (
                "I couldn't find enough matching movies. "
                "Could you provide more details?"
            )
        }

    # Build context — title + plot snippet
    context = "\n\n".join(
        f"{d.metadata.get('title', 'Unknown')} ({d.metadata.get('year', '')}): "
        f"{d.page_content[:400]}"
        for d in docs
    )

    # Tell LLM exactly which titles the KG confirmed are relevant
    kg_titles_str = "\n".join(f"- {t}" for t in kg_movies) if kg_movies else "N/A"

    prompt = f"""You are a helpful movie recommendation assistant.

User query: "{state['query']}"

The knowledge graph confirmed these movies match the query exactly:
{kg_titles_str}

Here are their plot summaries:
{context}

Instructions:
- Recommend ONLY the movies listed above — these are verified matches
- For each movie give the title, year, and a brief reason why it fits the query
- If plots don't match the query well, still trust the KG titles and explain what you know
- Be conversational and helpful
"""

    answer = llm_service.gemini_model.invoke(prompt).content
    logger.info(f"Final answer generated (len: {len(answer)})")
    return {"final_answer": answer}


def handle_invalid(state: MovieState) -> dict:
    """Handle invalid or gibberish queries — matches notebook."""
    logger.info("--- Node: Handle Invalid/Gibberish ---")

    message = (
        "I'm sorry, I didn't quite understand that. "
        "Could you please describe a movie plot, provide a title, "
        "or ask a specific question about actors or genres?"
    )

    logger.info(f"Invalid query handled: {state['query'][:50]}")
    return {"final_answer": message}
