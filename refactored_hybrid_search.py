"""
Refactored Hybrid Search Graph

Changes Made:
1. Removed KG handling from similarity_search - it now only does vector retrieval
2. Removed RELEVANCE_THRESHOLD from similarity_search - thresholding belongs in reranking
3. Added kg_results_to_docs node - converts KG movie titles to documents
4. Added hybrid_rerank node - reranks with semantic (0.7) + symbolic (0.3) scores
5. Updated graph edges - KG flow goes through kg_results_to_docs -> hybrid_rerank
6. Updated check_movie_exists and get_dataset_plot to use df1_cleaned

Copy the relevant code sections to your hybrid_search.ipynb notebook.
"""

# =============================================================================
# CELL: Imports and Setup (Replace your existing cell with ChatOllama/embeddings)
# =============================================================================

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma2:2b"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

movie_db = FAISS.load_local(
    "artifacts/movie_faiss_v3",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load cleaned dataset for movie existence checks
df1_cleaned = pd.read_csv('df1_cleaned.csv')


# =============================================================================
# CELL: check_movie_exists (Replace existing)
# =============================================================================

def check_movie_exists(state: MovieState):
    print(f"--- Node: Check Movie Exists ---")
    title = state["movie_name"]
    # Use df1_cleaned for existence check
    exists = df1_cleaned['Title'].str.lower().str.contains(title.lower(), na=False).any()
    print(f"Movie Exists in DB: {exists}")
    return {"movie_exists": exists}


# =============================================================================
# CELL: get_dataset_plot (Replace existing)
# =============================================================================

def get_dataset_plot(state: MovieState):
    print(f"--- Node: Get Dataset Plot ---")
    title = state["movie_name"]
    # Use df1_cleaned for plot retrieval
    mask = df1_cleaned['Title'].str.lower().str.contains(title.lower(), na=False)
    if mask.any():
        plot = df1_cleaned.loc[mask, 'Plot'].iloc[0]
        print("Retrieved plot from Dataset.")
        return {"base_plot": plot}
    print("Movie not found in Dataset.")
    return {"base_plot": None}


# =============================================================================
# CELL: similarity_search (Replace existing - SIMPLIFIED, no KG handling)
# =============================================================================

def similarity_search(state: MovieState):
    print(f"--- Node: Similarity Search ---")
    
    # Pure vector retrieval - no KG handling, no thresholding
    # Use the base_plot (from web/dataset) if available, otherwise the raw query
    query_text = state.get("base_plot") or state["query"]
    print(f"Performing semantic search for: {query_text[:50]}...")
    
    docs = movie_db.similarity_search(query_text, k=5)
    
    print(f"Total documents found: {len(docs)}")
    return {"final_docs": docs}


# =============================================================================
# CELL: kg_results_to_docs (NEW - Add after kg_agent)
# =============================================================================

def kg_results_to_docs(state: MovieState):
    print("--- Node: KG Results to Docs ---")

    titles = state.get("kg_movies", [])
    docs = []

    for title in titles:
        matches = movie_db.similarity_search(title, k=1)
        if matches:
            docs.append(matches[0])

    print(f"Converted {len(docs)} KG titles to documents.")
    return {"final_docs": docs}


# =============================================================================
# CELL: hybrid_rerank (NEW - Add after kg_results_to_docs)
# =============================================================================

def hybrid_rerank(state: MovieState):
    print("--- Node: Hybrid Rerank ---")

    docs = state.get("final_docs", [])
    if not docs:
        return {"final_docs": []}

    query_emb = embeddings.embed_query(state["query"])
    reranked = []

    for doc in docs:
        doc_emb = embeddings.embed_documents([doc.page_content])[0]
        semantic_score = cosine_similarity(
            [query_emb], [doc_emb]
        )[0][0]

        symbolic_score = 0.0
        if state.get("kg_movies") and doc.metadata.get("title") in state["kg_movies"]:
            symbolic_score += 1.0

        final_score = 0.7 * semantic_score + 0.3 * symbolic_score
        reranked.append((final_score, doc))

    reranked.sort(key=lambda x: x[0], reverse=True)
    print(f"Reranked to top {min(5, len(reranked))} documents.")
    return {"final_docs": [doc for _, doc in reranked[:5]]}


# =============================================================================
# CELL: Graph Definition (Replace existing graph definition cell)
# =============================================================================

# --- Routing Logic ---
def route_after_classification(state):
    intent = state.get("intent")
    if intent == "plot":
        return "similarity_search"  # Flow 1: Semantic Search
    elif intent == "movie_name":
        return "extract_movie_name" # Flows 2 & 3: Specific Name
    elif intent == "query_search":
        return "kg_agent"         # Flow 4: Query/KG Search
    elif intent == "invalid":
        return "handle_invalid"
    return "kg_agent"

def route_after_extraction(state):
    if state.get("movie_name"):
        return "check_movie_exists"
    else:
        # Fallback if classified as movie_name but no name found
        # Could go to plot search or end. Sending to plot search seems safest.
        return "similarity_search"

def route_movie_exists(state):
    if state["movie_exists"]:
        return "get_dataset_plot"  # Flow 2: Existing in Dataset
    return "get_web_plot"          # Flow 3: Web Search

# --- Graph Definition ---
graph = StateGraph(MovieState)

graph.add_node("classify_query", classify_query)
graph.add_node("extract_movie_name", extract_movie_name)
graph.add_node("check_movie_exists", check_movie_exists)
graph.add_node("get_dataset_plot", get_dataset_plot)
graph.add_node("get_web_plot", get_web_plot)
graph.add_node("kg_agent", kg_agent)
graph.add_node("kg_results_to_docs", kg_results_to_docs)  # NEW NODE
graph.add_node("hybrid_rerank", hybrid_rerank)             # NEW NODE
graph.add_node("similarity_search", similarity_search)
graph.add_node("generate_answer", generate_answer)
graph.add_node("handle_invalid", handle_invalid)
graph.set_entry_point("classify_query")

# Edge 1: Classify -> (Route)
graph.add_conditional_edges(
    "classify_query",
    route_after_classification,
    {
        "similarity_search": "similarity_search",
        "extract_movie_name": "extract_movie_name",
        "kg_agent": "kg_agent",
        "handle_invalid": "handle_invalid"
    }
)

# Edge 2: Extract Name -> (Route Check or Fallback)
graph.add_conditional_edges(
    "extract_movie_name",
    route_after_extraction,
    {
        "check_movie_exists": "check_movie_exists",
        "similarity_search": "similarity_search"
    }
)

# Edge 3: Check Exists -> (Route YES/NO)
graph.add_conditional_edges(
    "check_movie_exists",
    route_movie_exists,
    {
        "get_dataset_plot": "get_dataset_plot",
        "get_web_plot": "get_web_plot"
    }
)

# Plot paths -> similarity_search -> hybrid_rerank
graph.add_edge("get_dataset_plot", "similarity_search")
graph.add_edge("get_web_plot", "similarity_search")
graph.add_edge("similarity_search", "hybrid_rerank")

# KG flow -> kg_results_to_docs -> hybrid_rerank
graph.add_edge("kg_agent", "kg_results_to_docs")
graph.add_edge("kg_results_to_docs", "hybrid_rerank")

# Final Answer
graph.add_edge("hybrid_rerank", "generate_answer")
graph.add_edge("generate_answer", END)
graph.add_edge("handle_invalid", END)

app = graph.compile()
