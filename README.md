# PlotSense

PlotSense is a powerful movie search and recommendation engine that brings together semantic search, hybrid search capabilities, and intelligent querying using Large Language Models (LLMs). It features a robust Python FastAPI backend and a sleek, animated Next.js frontend for an immersive user experience.

## 🚀 Key Features

- **Knowledge Graph Integration:** Utilizes Neo4j for managing and querying structured graph data (like directors, actors, and genres).
- **Semantic & Hybrid Search:** Powered by FAISS and Python-based data pipelines to perform fast and accurate similarity searches based on plot descriptions.
- **Dynamic Poster Fetching:** Automatically searches and retrieves high-resolution movie posters concurrently using the IMDb public autocomplete API so your recommendations always look stunning.
- **Robust API Backend:** Fully functional REST API built with FastAPI and LangGraph for complex query routing.
- **Beautiful Frontend:** An interactive, beautifully animated Next.js UI featuring immersive full-screen movie detail modals.
- **LLM Integrations:** Out-of-the-box support for leading models via LangChain, utilizing Groq for blazing-fast inference.

## 📁 Repository Structure

- `backend/` - The core FastAPI backend logic, LangGraph workflows, and vector store integrations.
- `frontend/` - The Next.js application providing the intuitive Search UI.
- `requirements.txt` - Project python dependencies for the backend.
- `*.ipynb` - Assorted Jupyter Notebooks for building knowledge graphs, FAISS indices, exploring data, and standalone semantic search testing.

## 🛠️ Installation & Setup

1. **Set up a Virtual Environment (Backend):**
   It is recommended to use an isolated environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Backend Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have your `.env` variables containing your LLM (Groq/Tavily) and Neo4j API keys set up in the root directory.*

3. **Install Frontend Dependencies:**
   ```bash
   cd frontend
   npm install
   ```

## 🏃 Running the Application

You will need two terminal windows actively running to use PlotSense:

**1. Start the Backend API:**
```bash
cd backend
python3 main.py
```
*The API will be accessible at `http://0.0.0.0:8000`.*

**2. Start the Frontend UI:**
```bash
cd frontend
npm run dev
```
*The web interface will be accessible at `http://localhost:3000`.*

## 🧪 Experiments

If you want to view the experimental setups, check out the various interactive Jupyter notebooks (like `hybrid_search_verbose.ipynb`, and `semantic_search.ipynb`) in the root directory.