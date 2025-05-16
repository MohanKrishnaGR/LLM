# LLM/config.py
# Configuration settings for the LLM application

# LLM Model Identifiers
LLAMA4_MODEL = "llama3-70b-8192"  # Identifier for Llama3 70b model
DEEPSEEK_MODEL = "deepseek-coder-33b-instruct"  # Identifier for DeepSeek Coder 33b model

# Embedding Model Identifier
NOMIC_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # Identifier for Nomic embedding model

# Data and Index Directories
DATA_DIR = "data_for_rag"  # Directory to store documents for RAG
INDEX_DIR_PREFIX = "llm_index_"  # Prefix for vector store index directories
GRAPH_INDEX_DIR_PREFIX = "llm_graph_index_" # Prefix for knowledge graph index directories

# Re-ranker Model Identifier
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Identifier for the re-ranking model

# Query Transformation Prompt (Example)
DEFAULT_QUERY_TRANSFORMATION_PROMPT = (
    "Given the following user query, rephrase it and expand it with related concepts "
    "to improve information retrieval. Return only the transformed query.\n"
    "Original Query: {original_query}\n"
    "Transformed Query: "
)

# You can add other global configurations here, for example:
# DEFAULT_TEMPERATURE = 0.7
# DEFAULT_MAX_TOKENS = 2048