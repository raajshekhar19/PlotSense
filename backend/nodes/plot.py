"""
Plot Retrieval Node for PlotSense backend.
Gets movie plots from dataset — matches notebook.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from models import MovieState
from config import DATASET_PATH
from logger import get_logger

logger = get_logger(__name__)

# Load the local dataset into memory once using config path
try:
    df1_cleaned = pd.read_csv(DATASET_PATH)
    logger.info(f"plot.py: Dataset loaded from {DATASET_PATH} ({len(df1_cleaned)} records)")
except FileNotFoundError:
    logger.error(f"Could not find dataset at {DATASET_PATH}")
    df1_cleaned = pd.DataFrame(columns=['Title', 'Plot'])

def get_dataset_plot(state: MovieState) -> dict:
    """
    Get the movie plot from the local dataset using pandas.
    """
    logger.info("--- Node: Get Dataset Plot ---")
    
    title = state.get("movie_name", "")
    
    if not title:
        return {"base_plot": None}

    mask = df1_cleaned['Title'].str.lower().str.contains(title.lower(), na=False)
    
    if mask.any():
        plot = df1_cleaned.loc[mask, 'Plot'].iloc[0]
        logger.info(f"Retrieved plot from Dataset for {title}.")
        return {"base_plot": plot}
        
    logger.warning("Movie not found in Dataset.")
    return {"base_plot": None}
