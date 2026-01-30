import json
import os

NOTEBOOK_PATH = r"d:/emerging/hybrid_search.ipynb"

def refactor_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Helper to find cell by unique content snippet
    def find_cell_index(snippet):
        for i, cell in enumerate(nb['cells']):
            source = "".join(cell['source'])
            if snippet in source:
                return i
        return -1

    # 1. Update Classify Query Node
    # Intent mapping:
    # 1. "plot" -> Semantic Movie Search Path
    # 2. "movie_name" -> Specific Movie Name Path
    # 3. "query_search" -> AI Agent/KG Path
    idx_classify = find_cell_index("def classify_query(state: MovieState):")
    if idx_classify != -1:
        nb['cells'][idx_classify]['source'] = [
            "def classify_query(state: MovieState):\n",
            "    print(f\"--- Node: Classify Query ---\")\n",
            "    prompt = f\"\"\"\n",
            "Classify the query strictly into one of the following intents:\n",
            "1. plot: User describes a plot (Semantic Movie Search Flow).\n",
            "2. movie_name: User provides a specific movie title (Specific Movie Name Flow).\n",
            "3. query_search: Complex query or general question requiring AI interpretation (Query Movie Search Flow).\n",
            "\n",
            "Query:\n",
            "{state['query']}\n",
            "\"\"\"\n",
            "    intent = llm.invoke(prompt).content.strip().lower()\n",
            "    # Basic normalization\n",
            "    if \"plot\" in intent: intent = \"plot\"\n",
            "    elif \"movie\" in intent and \"name\" in intent: intent = \"movie_name\"\n",
            "    elif \"query\" in intent or \"complex\" in intent: intent = \"query_search\"\n",
            "    else: intent = \"query_search\" # Default fallback\n",
            "    \n",
            "    print(f\"Identified Intent: {intent}\")\n",
            "    return {\"intent\": intent}\n"
        ]

    # 2. Update Extract Movie Name Node
    # Handle "NONE" case or empty extraction
    idx_extract = find_cell_index("def extract_movie_name(state: MovieState):")
    if idx_extract != -1:
        nb['cells'][idx_extract]['source'] = [
            "def extract_movie_name(state: MovieState):\n",
            "    print(f\"--- Node: Extract Movie Name ---\")\n",
            "    prompt = f\"\"\"\n",
            "Extract the specific movie name from the query.\n",
            "If no specific movie name is found, respond EXACTLY with \"NONE\".\n",
            "Query:\n",
            "{state['query']}\n",
            "\"\"\"\n",
            "    name = llm.invoke(prompt).content.strip()\n",
            "    print(f\"Extracted Name: {name}\")\n",
            "    if \"NONE\" in name or not name:\n",
            "        return {\"movie_name\": None}\n",
            "    return {\"movie_name\": name}\n"
        ]

    # 3. Update KG Agent Node (Query Movie Search Flow)
    # Ensure it's labeled correctly and feeds into similarity search
    # This seems generally okay, but let's check the keys.
    # User's graph: Query -> Classify -> Query movie search -> AI Agent -> Derived from dataset using Knowledge Graph -> Semantic search on plots -> Final rankings.
    # Current code: kg_agent extracts entities -> run_cypher -> returns kg_movies.
    # Implementation matches "derived from dataset...".

    # 4. Update Graph Topology (Route Definitions and Edges)
    idx_graph = find_cell_index("graph = StateGraph(MovieState)")
    if idx_graph != -1:
        nb['cells'][idx_graph]['source'] = [
            "# --- Routing Logic ---\n",
            "def route_after_classification(state):\n",
            "    intent = state.get(\"intent\")\n",
            "    if intent == \"plot\":\n",
            "        return \"similarity_search\"  # Flow 1: Semantic Search\n",
            "    elif intent == \"movie_name\":\n",
            "        return \"extract_movie_name\" # Flows 2 & 3: Specific Name\n",
            "    elif intent == \"query_search\":\n",
            "        return \"kg_agent\"         # Flow 4: Query/KG Search\n",
            "    return \"kg_agent\"\n",
            "\n",
            "def route_after_extraction(state):\n",
            "    if state.get(\"movie_name\"):\n",
            "        return \"check_movie_exists\"\n",
            "    else:\n",
            "        # Fallback if classified as movie_name but no name found\n",
            "        # Could go to plot search or end. Sending to plot search seems safest.\n",
            "        return \"similarity_search\"\n",
            "\n",
            "def route_movie_exists(state):\n",
            "    if state[\"movie_exists\"]:\n",
            "        return \"get_dataset_plot\"  # Flow 2: Existing in Dataset\n",
            "    return \"get_web_plot\"          # Flow 3: Web Search\n",
            "\n",
            "# --- Graph Definition ---\n",
            "graph = StateGraph(MovieState)\n",
            "\n",
            "graph.add_node(\"classify_query\", classify_query)\n",
            "graph.add_node(\"extract_movie_name\", extract_movie_name)\n",
            "graph.add_node(\"check_movie_exists\", check_movie_exists)\n",
            "graph.add_node(\"get_dataset_plot\", get_dataset_plot)\n",
            "graph.add_node(\"get_web_plot\", get_web_plot)\n",
            "graph.add_node(\"kg_agent\", kg_agent)\n",
            "graph.add_node(\"similarity_search\", similarity_search)\n",
            "graph.add_node(\"generate_answer\", generate_answer)\n",
            "\n",
            "graph.set_entry_point(\"classify_query\")\n",
            "\n",
            "# Edge 1: Classify -> (Route)\n",
            "graph.add_conditional_edges(\n",
            "    \"classify_query\",\n",
            "    route_after_classification,\n",
            "    {\n",
            "        \"similarity_search\": \"similarity_search\",\n",
            "        \"extract_movie_name\": \"extract_movie_name\",\n",
            "        \"kg_agent\": \"kg_agent\"\n",
            "    }\n",
            ")\n",
            "\n",
            "# Edge 2: Extract Name -> (Route Check or Fallback)\n",
            "graph.add_conditional_edges(\n",
            "    \"extract_movie_name\",\n",
            "    route_after_extraction,\n",
            "    {\n",
            "        \"check_movie_exists\": \"check_movie_exists\",\n",
            "        \"similarity_search\": \"similarity_search\"\n",
            "    }\n",
            ")\n",
            "\n",
            "# Edge 3: Check Exists -> (Route YES/NO)\n",
            "graph.add_conditional_edges(\n",
            "    \"check_movie_exists\",\n",
            "    route_movie_exists,\n",
            "    {\n",
            "        \"get_dataset_plot\": \"get_dataset_plot\",\n",
            "        \"get_web_plot\": \"get_web_plot\"\n",
            "    }\n",
            ")\n",
            "\n",
            "# Convergence to Similarity Search\n",
            "graph.add_edge(\"get_dataset_plot\", \"similarity_search\")\n",
            "graph.add_edge(\"get_web_plot\", \"similarity_search\")\n",
            "graph.add_edge(\"kg_agent\", \"similarity_search\")\n",
            "\n",
            "# Final Answer\n",
            "graph.add_edge(\"similarity_search\", \"generate_answer\")\n",
            "graph.add_edge(\"generate_answer\", END)\n",
            "\n",
            "app = graph.compile()\n"
        ]

        # Note: Original code had 'route_after_classification' and 'route_movie_exists' defined in separate cells.
        # I am combining the routing logic into the graph definition cell for clarity and to ensure they are updated.
        # I should clear the old cells or leave them (they won't be used by the new app compile).
        # Better: clear or overwrite the old routing function cells to avoid confusion.

    idx_route1 = find_cell_index("def route_after_classification(state):")
    if idx_route1 != -1 and idx_route1 != idx_graph:
         nb['cells'][idx_route1]['source'] = ["# Merged into Graph Definition Cell\n"]

    idx_route2 = find_cell_index("def route_movie_exists(state):")
    if idx_route2 != -1 and idx_route2 != idx_graph:
         nb['cells'][idx_route2]['source'] = ["# Merged into Graph Definition Cell\n"]

    # 5. Fix Similarity Search explanation log
    idx_sim = find_cell_index("def similarity_search(state: MovieState):")
    if idx_sim != -1:
         nb['cells'][idx_sim]['source'] = [
            "def similarity_search(state: MovieState):\n",
            "    print(f\"--- Node: Similarity Search ---\")\n",
            "    # If we have a list of movies from KG, prioritize searching for them\n",
            "    if state.get(\"kg_movies\"):\n",
            "        print(\"Searching based on KG movie list.\")\n",
            "        docs = []\n",
            "        for title in state[\"kg_movies\"]:\n",
            "            docs.extend(movie_db.similarity_search(title, k=2))\n",
            "    else:\n",
            "        print(\"Searching based on Plot Embeddings (Top 5 Extract).\")\n",
            "        # If base_plot is retrieved (from DB or Web), use that. Else use original query.\n",
            "        query_text = state.get(\"base_plot\") or state[\"query\"]\n",
            "        docs = movie_db.similarity_search(query_text, k=5)\n",
            "\n",
            "    return {\"final_docs\": docs}\n"
         ]

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Refactoring complete.")

if __name__ == "__main__":
    refactor_notebook()
