"""
Configuration settings for the RAG application.
Contains model settings, directory paths, and application parameters.
"""

import os

# Model configurations
LLAMA4_MODEL = "llama3-70b-8192"
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"

# Directory configurations
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data_for_rag")
PERSIST_DIR_BASE = os.path.join(PROJECT_ROOT, "persisted_index")
DOCUMENT_TRACKING_FILE = os.path.join(PERSIST_DIR_BASE, "document_states.json")

# Supported file types
SUPPORTED_FILE_EXTENSIONS = [
    ".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"
]

# Model identifiers
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"
JUDGE_MODEL_NAME = LLAMA4_MODEL

# Default processing parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 20

# Query transformation template
DEFAULT_QUERY_TRANSFORMATION_PROMPT = (
    "Given the following user query, rephrase it and expand it with related concepts "
    "to improve information retrieval. Return only the transformed query.\n"
    "Original Query: {original_query}\n"
    "Transformed Query: "
)

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
