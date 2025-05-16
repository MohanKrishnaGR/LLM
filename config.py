# LLM/config.py
import os

# Model configurations
LLAMA4_MODEL = "llama3-70b-8192"
DEEPSEEK_MODEL = "deepseek-coder-33b-instruct"

# --- IMPORTANT DIRECTORY CONFIGURATIONS ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_for_rag")
PERSIST_DIR_BASE = os.path.join(PROJECT_ROOT, "persisted_index")
DOCUMENT_TRACKING_FILE = os.path.join(PERSIST_DIR_BASE, "document_states.json")
# --- END IMPORTANT DIRECTORY CONFIGURATIONS ---

# Embedding model
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2" # Example

# Re-ranker model
RERANKER_MODEL_ID = "BAAI/bge-reranker-base" # Ensure this line exists and is correct
# RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2" # Alternative from original user files

# LLM as a Judge model
JUDGE_MODEL_NAME = LLAMA4_MODEL

# Other settings
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 20

# Query Transformation Prompt (Example)
DEFAULT_QUERY_TRANSFORMATION_PROMPT = (
    "Given the following user query, rephrase it and expand it with related concepts "
    "to improve information retrieval. Return only the transformed query.\n"
    "Original Query: {original_query}\n"
    "Transformed Query: "
)

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# API Keys - To be managed by user (e.g., via Streamlit secrets or environment variables)
# Example:
# GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
