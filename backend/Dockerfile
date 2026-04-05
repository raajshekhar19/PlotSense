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
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

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

RUN pip install gdown

RUN gdown --id 13cya6zHklhYAuXX8DDqscU4ftCbKA8cU -O /app/faiss_index.zip && \
    unzip /app/faiss_index.zip -d /app/ && \
    rm /app/faiss_index.zip

RUN gdown --id DATASET_FILE_ID -O /app/dataset.csv

# Copy application source code
COPY . .

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
