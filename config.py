# LLM/config.py
import os
from pathlib import Path

# --- Existing configurations ---
# Example: Assuming ROOT_DIR is the 'LLM' directory for this example
# Adjust ROOT_DIR if your config.py is elsewhere relative to the project structure
ROOT_DIR = Path(__file__).parent.resolve() # LLM directory

# LLM and Embedding Model Configurations
LLAMA4_MODEL = "llama3-70b-8192"
DEEPSEEK_MODEL = "deepseek-coder-33b-instruct" # Example, adjust if needed
NOMIC_EMBED_MODEL = "nomic-embed-text" # Keep this if it's your preferred default
DEFAULT_EMBED_MODEL_NAME = NOMIC_EMBED_MODEL
DEFAULT_LLM_MODEL_NAME = LLAMA4_MODEL

# Data and Indexing Configurations
DATA_DIR = str(ROOT_DIR / "data_for_rag") # Make sure this points to your data
PROMPT_TEMPLATE_STR = """
# ... (your existing prompt template) ...
"""
DEFAULT_QUERY_TRANSFORMATION_PROMPT = """
# ... (your existing query transformation prompt) ...
"""

# --- New configurations for Persistence and Incremental Processing ---
# Base directory for all persistent data related to indexing
PERSIST_APP_DATA_DIR = ROOT_DIR / "app_data"
PERSIST_INDEX_BASE_DIR = PERSIST_APP_DATA_DIR / "persisted_indices"

# Specific persistence paths for different index types
VECTOR_INDEX_PERSIST_DIR = str(PERSIST_INDEX_BASE_DIR / "vector_index")
KG_INDEX_PERSIST_DIR = str(PERSIST_INDEX_BASE_DIR / "kg_index")

# Path for storing metadata about processed files
PROCESSED_FILES_METADATA_PATH = str(PERSIST_APP_DATA_DIR / "processed_files.json")

# Knowledge Graph specific build configurations
KG_MAX_TRIPLETS_PER_CHUNK = 5
KG_INCLUDE_EMBEDDINGS = True

# Ensure data directory exists (it's good practice, though your app might create it)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# --- Ensure base persistence directories exist ---
Path(VECTOR_INDEX_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(KG_INDEX_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_FILES_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)

# You might want to move UI related defaults from here to a UI config or keep them
# For example, if you had UI defaults like:
# DEFAULT_TOP_K = 3
# DEFAULT_GRAPH_RETRIEVER_MODE = "keyword"