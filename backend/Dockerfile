# =============================================================================
# IMPORTANT: Before building this Docker image, make sure the FAISS index files
# are present in backend/faiss_index/ and dataset.csv is in backend/.
# These files are NOT in the GitHub repo and must be copied manually before
# each build.
#
# Run the following commands from the project root:
#   cp -r artifacts/movie_faiss_v3/ backend/faiss_index/
#   cp dataset.csv backend/dataset.csv
#
# Then build:
#   docker build -t plotsense-backend ./backend
# =============================================================================

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr (important for Docker logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Copy FAISS index files into the image (baked in — no persistent volume needed)
# These must exist at build time in backend/faiss_index/
COPY faiss_index/ /app/faiss_index/

# Copy dataset into the image
COPY dataset.csv /app/dataset.csv

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
