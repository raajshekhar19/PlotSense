# PlotSense Movie Search Architecture and Main Logic

The PlotSense backend is a sophisticated, hybrid semantic search engine designed to return highly relevant movie recommendations based on natural language queries. It leverages **LangGraph** to manage a complex retrieval workflow, integrating a **Neo4j Knowledge Graph**, **FAISS Vector Store**, **Tavily Web Search**, and a **Cross-Encoder Reranker** to handle a variety of query types efficiently.

Below is an extremely detailed breakdown of the system architecture and its core workflow logic.

## 1. System Components (Services)

The system is built on a loosely coupled microservice-like structure within the FastAPI application. Each external integration is managed by a distinct service module.

- **FastAPI Endpoints (`main.py`)**: The primary entry point. It handles HTTP requests, CORS, and provides APIs for search (`/search`) and health checks (`/health` and `/health/services`). It also manages the lifecycle of the services, initializing them at startup.
- **LLM Service (`llm_service.py`)**: Manages interactions with the Large Language Model (e.g., Llama, GPT). Used for query classification, information extraction, final answer generation, and Cypher query generation for Neo4j.
- **Neo4j Service (`neo4j_service.py`)**: Interfaces with a Neo4j Knowledge Graph. This is extremely useful for structured data queries (e.g., finding all movies by a specific director, or actors who starred together in a specific year).
- **FAISS Service (`faiss_service.py`)**: Manages the FAISS vector database. Used for dense similarity search (semantic matching) against movie plots embedded in a vector space. Ideal for "movies where a man goes to mars and grows potatoes" type queries.
- **Tavily Service (`tavily_service.py`)**: Acts as an online web fallback mechanism. If a user asks for "movies like [X]" and movie [X] doesn't exist in the local dataset, Tavily is queried to fetch the plot of [X] from the web so the similarity search can still proceed.
- **Reranker Service (`reranker_service.py`)**: Uses a Cross-Encoder model to analyze and re-score the retrieved documents against the user's query. This drastically improves precision over a pure dual-encoder (FAISS) retrieval.

---

## 2. Core Workflow Logic (LangGraph)

The core strength of the architecture is its dynamic state graph (implemented in `graph.py`), which routes queries intelligently based on their inferred intent and the success of intermediate steps.

### A. Query Classification (`classify_query`)
When a request enters the workflow, an LLM attempts to classify the intent into one of four categories:
1. **`plot`**: The user is describing a movie plot (e.g., "guy who gets stuck on mars").
2. **`movie_name`**: The user is looking for movies similar to a given title (e.g., "movies like The Matrix").
3. **`query_search`**: The user is asking a structured/complex query (e.g., "movies directed by Christopher Nolan in the 2000s").
4. **`invalid`**: Off-topic queries (routed to `handle_invalid`).

### B. Routing and Execution Paths

#### Path 1: Plot Intent (`plot`)
If the query describes a plot, it is sent directly to `similarity_search` (FAISS) to fetch semantically similar plot descriptions from the vector database.

#### Path 2: Movie Name Intent (`movie_name`)
If the user specifies "movies like [X]":
1. **`extract_movie_name`**: An LLM extracts the exact movie name from the query.
2. **`check_movie_exists`**: The system checks if this movie exists in the local database.
    - **Local Match (`get_dataset_plot`)**: If it exists, the plot is extracted from the local dataset and forwarded to `similarity_search` to find matching movies in the FAISS index.
    - **No Local Match (`get_web_plot`)**: If the movie isn't in the dataset, the `tavily_service` is queried to fetch the plot from the internet.
        - *Success*: If a usable plot is fetched, it moves to `similarity_search`.
        - *Failure/Ambiguity (`clarification_node`)*: If the web search fails or the movie is too ambiguous, the system halts the search and asks the user for clarification.

#### Path 3: Complex Queries (`query_search`)
For structured attributes (directors, actors, cast, dates):
1. **`kg_agent`**: A Cypher agent converts the natural language query into a Cypher query and executes it against the Neo4j Knowledge Graph.
2. **Evaluation**:
    - **Dense Results (`kg_results_to_docs`)**: If Neo4j returns rich results, they are converted into document objects and sent to the reranking stage.
    - **Sparse Results (`kg_web_fallback`)**: If Neo4j returns very few or no results (e.g., the knowledge graph doesn't have that specific data), the system falls back to Tavily web search.
        - *Web Content Found (`generate_from_web`)*: An answer is generated directly from the web results.
        - *No Web Content (`similarity_search`)*: As a last resort, it falls back to a semantic search against the FAISS index.

### C. Reranking (`hybrid_rerank`)
Regardless of whether documents were retrieved via FAISS (`similarity_search`) or Neo4j (`kg_results_to_docs`), the initial retrieval results are passed through a Cross-Encoder reranker. The reranker computes an interaction score between the original user query and each retrieved document, reordering them so the most contextually relevant documents are pushed to the top.

### D. Final Answer Generation (`generate_answer`)
The top-K reranked documents and the original query are passed to an LLM. The LLM synthesizes a final, human-readable answer (including metadata like movie names, release years, and directors), directly addressing the user's intent.

---

## 3. Data Flow Summary

```text
User Request (FastAPI) 
      │
      ▼
LangGraph Classifier ──► (Invalid) ──► Error Message
      │
      ├──► (Plot) ─► FAISS Search ─┐
      │                            │
      ├──► (Movie Like X)          │
      │       │                    │
      │       ├─► Check DB ─┬─► In DB ──► Fetch Plot ───┐
      │       │             └─► Not In DB ─► Web Search ┴─► FAISS Search ─┐
      │       v                                                           │
      │     (Clarification Question needed if web fails)                  │
      │                                                                   │
      └──► (Complex Query)                                                │
              │                                                           │
              ├─► Neo4j Agent ─┬─► Graph Found Data ──────────────────────┤
              │                └─► Sparse Data ─► Web Fallback            │
              │                                      ├─► Web Answer ──► (End)
              │                                      └─► FAISS Search ────┤
              ▼                                                           ▼
                                                                  Cross-Encoder Reranker
                                                                          │
                                                                          ▼
                                                                     LLM Synthesis
                                                                          │
                                                                          ▼
                                                            API Response (Frontend UI)
```

## 4. Strengths of the Architecture

- **Fallbacks & Resilience**: The graph structure ensures that if one service fails or lacks data (e.g., a movie isn't in the database or the Knowledge graph is missing an edge), the system elegantly degrades to web search or similarity search rather than throwing an error.
- **High Precision**: The combination of Neo4j for exact relationship queries and FAISS + Cross-Encoder for dense vector similarity gives the best of both structured and unstructured search paradims.
- **Extensibility**: By using LangGraph nodes, adding new capabilities (like querying streaming services to see "where to watch") is as simple as adding a new node and edge.
