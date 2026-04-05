"""
LangGraph Workflow Definition for PlotSense backend.
Full graph rewrite matching hybrid_search_verbose.ipynb.
"""
from langgraph.graph import StateGraph, END

from models import MovieState
from nodes.classify import classify_query
from nodes.extract import extract_movie_name, check_movie_exists, extract_kg_entities_node
from nodes.plot import get_dataset_plot
from nodes.search import get_web_plot, similarity_search, generate_from_web, strip_intent_language
from nodes.kg import cypher_agent, kg_results_to_docs, kg_web_fallback, check_entity_exists, handle_missing_entity
from nodes.rerank import hybrid_rerank
from nodes.generate import generate_answer, handle_invalid
from nodes.clarify import clarification_node
from logger import get_logger

logger = get_logger(__name__)


# =====================
# Routing Functions
# =====================

def route_after_classification(state: MovieState):
    intent = state.get("intent")
    logger.info(f"Routing: intent={intent}")
    if intent == "plot":
        return "similarity_search"
    elif intent == "movie_name":
        return "extract_movie_name"
    elif intent == "query_search":
        return "extract_kg_entities_node"
    elif intent == "invalid":
        return "handle_invalid"
    return "extract_kg_entities_node"


def route_after_extraction(state: MovieState):
    if state.get("movie_name"):
        return "check_movie_exists"
    return "similarity_search"


def route_movie_exists(state: MovieState):
    if state["movie_exists"]:
        return "get_dataset_plot"
    return "get_web_plot"


def route_after_web_plot(state: MovieState):
    """If web returned useful content → proceed. Otherwise → ask for clarification."""
    base_plot = state.get("base_plot", "")
    if base_plot and len(base_plot) > 100:
        return "similarity_search"
    return "clarification_node"


def route_after_entity_check(state: MovieState):
    if state.get("entity_not_found"):
        return "handle_missing_entity"
    return "kg_agent"


def route_after_kg(state: MovieState):
    if state.get("kg_sparse"):
        entities = state.get("kg_entities", {})
        if entities.get("actor") or entities.get("director"):
            return "handle_missing_entity"
        return "kg_web_fallback"
    return "kg_results_to_docs"


def route_after_kg_web_fallback(state: MovieState):
    if state.get("web_context"):
        return "generate_from_web"
    return "strip_intent_language"


def route_after_missing_entity(state: MovieState):
    if state.get("search_status") == "not_found":
        return END
    return "generate_from_web"


# =====================
# Graph Building
# =====================

def build_graph():
    """Build and compile the full LangGraph workflow."""
    logger.info("Building LangGraph workflow (hybrid_search_verbose edition)...")

    graph = StateGraph(MovieState)

    # ── Nodes ───────────────────────────────────────────────────
    graph.add_node("classify_query",     classify_query)
    graph.add_node("extract_movie_name", extract_movie_name)
    graph.add_node("check_movie_exists", check_movie_exists)
    graph.add_node("get_dataset_plot",   get_dataset_plot)
    graph.add_node("get_web_plot",       get_web_plot)
    graph.add_node("clarification_node", clarification_node)
    
    # New KG Nodes
    graph.add_node("extract_kg_entities_node", extract_kg_entities_node)
    graph.add_node("check_entity_exists",      check_entity_exists)
    graph.add_node("handle_missing_entity",    handle_missing_entity)
    graph.add_node("strip_intent_language",    strip_intent_language)
    
    graph.add_node("kg_agent",           cypher_agent)
    graph.add_node("kg_results_to_docs", kg_results_to_docs)
    graph.add_node("kg_web_fallback",    kg_web_fallback)
    graph.add_node("generate_from_web",  generate_from_web)
    graph.add_node("hybrid_rerank",      hybrid_rerank)
    graph.add_node("similarity_search",  similarity_search)
    graph.add_node("generate_answer",    generate_answer)
    graph.add_node("handle_invalid",     handle_invalid)

    # ── Entry ────────────────────────────────────────────────────
    graph.set_entry_point("classify_query")

    # ── Conditional edges ────────────────────────────────────────
    graph.add_conditional_edges("classify_query", route_after_classification, {
        "similarity_search":          "similarity_search",
        "extract_movie_name":         "extract_movie_name",
        "extract_kg_entities_node":   "extract_kg_entities_node",
        "handle_invalid":             "handle_invalid"
    })

    graph.add_conditional_edges("extract_movie_name", route_after_extraction, {
        "check_movie_exists": "check_movie_exists",
        "similarity_search":  "similarity_search"
    })

    graph.add_conditional_edges("check_movie_exists", route_movie_exists, {
        "get_dataset_plot": "get_dataset_plot",
        "get_web_plot":     "get_web_plot"
    })

    # After web plot: check if we found useful content or need clarification
    graph.add_conditional_edges("get_web_plot", route_after_web_plot, {
        "similarity_search":  "similarity_search",
        "clarification_node": "clarification_node"
    })
    
    # Entity Check Conditional
    graph.add_conditional_edges("check_entity_exists", route_after_entity_check, {
        "handle_missing_entity": "handle_missing_entity",
        "kg_agent":              "kg_agent"
    })

    # KG agent: sparse → web fallback OR missing entity handler, else → docs
    graph.add_conditional_edges("kg_agent", route_after_kg, {
        "handle_missing_entity": "handle_missing_entity",
        "kg_web_fallback":       "kg_web_fallback",
        "kg_results_to_docs":    "kg_results_to_docs"
    })

    # KG web fallback: got content → generate directly, else → strip language
    graph.add_conditional_edges("kg_web_fallback", route_after_kg_web_fallback, {
        "generate_from_web":     "generate_from_web",
        "strip_intent_language": "strip_intent_language"
    })
    
    # Missing entity handler: web found -> generate, web failed -> END
    graph.add_conditional_edges("handle_missing_entity", route_after_missing_entity)

    # ── Fixed edges ──────────────────────────────────────────────
    graph.add_edge("extract_kg_entities_node", "check_entity_exists")
    graph.add_edge("strip_intent_language",    "similarity_search")
    
    graph.add_edge("get_dataset_plot",   "similarity_search")
    graph.add_edge("similarity_search",  "hybrid_rerank")
    graph.add_edge("kg_results_to_docs", "hybrid_rerank")
    graph.add_edge("hybrid_rerank",      "generate_answer")
    graph.add_edge("generate_answer",    END)
    graph.add_edge("generate_from_web",  END)
    graph.add_edge("clarification_node", END)
    graph.add_edge("handle_invalid",     END)

    app = graph.compile()
    logger.info("Workflow compiled successfully")
    return app


# =====================
# Singleton
# =====================

workflow_app = None


def get_workflow():
    """Get or build the compiled workflow."""
    global workflow_app
    if workflow_app is None:
        workflow_app = build_graph()
    return workflow_app
