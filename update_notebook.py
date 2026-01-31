import json
import os

# Define the file path
notebook_path = r'd:\emerging\hybrid_search.ipynb'

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the new code content
imports_code = """
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from langchain_neo4j import Neo4jGraph
import os
from dotenv import  load_dotenv
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
neo4j = Neo4jGraph(url=NEO4J_URI,username=NEO4J_USERNAME,password=NEO4J_PASSWORD,refresh_schema=False)
neo4j
"""

# New setup code with pandas
setup_code = """
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
"""

check_movie_exists_code = """
def check_movie_exists(state: MovieState):
    print(f"--- Node: Check Movie Exists ---")
    title = state["movie_name"]
    # Use df1_cleaned for existence check
    exists = df1_cleaned['Title'].str.lower().str.contains(title.lower(), na=False).any()
    print(f"Movie Exists in DB: {exists}")
    return {"movie_exists": exists}
"""

get_dataset_plot_code = """
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
"""

similarity_search_code = """
def similarity_search(state: MovieState):
    print(f"--- Node: Similarity Search ---")
    
    # Pure vector retrieval - no KG handling, no thresholding
    # Use the base_plot (from web/dataset) if available, otherwise the raw query
    query_text = state.get("base_plot") or state["query"]
    print(f"Performing semantic search for: {query_text[:50]}...")
    
    docs = movie_db.similarity_search(query_text, k=5)
    
    print(f"Total documents found: {len(docs)}")
    return {"final_docs": docs}
"""

kg_results_to_docs_code = """
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
"""

hybrid_rerank_code = """
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
"""

graph_code = """
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
graph.add_node("kg_results_to_docs", kg_results_to_docs)
graph.add_node("hybrid_rerank", hybrid_rerank)
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
"""

# Helper to create a code cell
def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

# Function to find cell index by content substring (naive but should work)
def find_cell_index(notebook, substring):
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if substring in source:
                return i
    return -1

# 1. Update Imports/Setup
idx = find_cell_index(nb, "ChatOllama")
if idx != -1:
    nb['cells'][idx] = create_code_cell(setup_code)
    print("Updated Setup/Imports cell.")
else:
    print("Warning: Could not find Setup/Imports cell.")

# 2. Update check_movie_exists
idx = find_cell_index(nb, "def check_movie_exists")
if idx != -1:
    nb['cells'][idx] = create_code_cell(check_movie_exists_code)
    print("Updated check_movie_exists cell.")
else:
    print("Warning: Could not find check_movie_exists cell.")

# 3. Update get_dataset_plot
idx = find_cell_index(nb, "def get_dataset_plot")
if idx != -1:
    nb['cells'][idx] = create_code_cell(get_dataset_plot_code)
    print("Updated get_dataset_plot cell.")
else:
    print("Warning: Could not find get_dataset_plot cell.")

# 4. Update similarity_search
idx = find_cell_index(nb, "def similarity_search")
similarity_search_idx = idx
if idx != -1:
    nb['cells'][idx] = create_code_cell(similarity_search_code)
    print("Updated similarity_search cell.")
else:
    print("Warning: Could not find similarity_search cell.")

# 5. Insert kg_results_to_docs and hybrid_rerank
# Find kg_agent to insert after it
kg_agent_idx = find_cell_index(nb, "def kg_agent")
if kg_agent_idx != -1:
    # Insert after kg_agent
    nb['cells'].insert(kg_agent_idx + 1, create_code_cell(kg_results_to_docs_code))
    nb['cells'].insert(kg_agent_idx + 2, create_code_cell(hybrid_rerank_code))
    print("Inserted kg_results_to_docs and hybrid_rerank cells.")
else:
    # If kg_agent not found, try inserting before similarity_search if available
    if similarity_search_idx != -1:
         nb['cells'].insert(similarity_search_idx, create_code_cell(kg_results_to_docs_code))
         nb['cells'].insert(similarity_search_idx + 1, create_code_cell(hybrid_rerank_code))
         print("Inserted kg_results_to_docs and hybrid_rerank cells (before similarity_search).")
    else:
        print("Warning: Could not find locaton to insert new nodes.")

# 6. Update Graph Definition
idx = find_cell_index(nb, "graph = StateGraph(MovieState)")
if idx != -1:
    nb['cells'][idx] = create_code_cell(graph_code)
    print("Updated graph definition cell.")
else:
    print("Warning: Could not find graph definition cell.")


# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook update complete.")
