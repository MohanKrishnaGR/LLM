import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from config import NOMIC_EMBED_MODEL
import logging
from typing import Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_api_key(api_key: str) -> bool:
    """Validate the API key format."""
    if not api_key or not isinstance(api_key, str):
        return False
    return api_key.strip() != ""

def get_llm(model_name: str, api_key: str) -> Optional[Any]:
    """
    Initialize and return a Groq LLM instance.
    
    Args:
        model_name: Name of the model to use
        api_key: Groq API key
        
    Returns:
        Optional[Any]: Configured LLM instance or None if initialization fails
    """
    if not validate_api_key(api_key):
        logger.error("Invalid or missing API key")
        st.error("Please provide a valid API key in the sidebar.")
        return None

    try:
        # Configure LLM with optimized settings
        llm = Groq(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9
        )
        logger.info(f"Successfully initialized LLM model: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"LLM initialization failed: {str(e)}")
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

def get_embedding_model() -> Optional[Any]:
    """
    Initialize and return the embedding model instance.
    
    Returns:
        Optional[Any]: Configured embedding model or None if initialization fails
    """
    try:
        # Configure embedding model with optimized settings
        embed_model = FastEmbedEmbedding(
            model_name=NOMIC_EMBED_MODEL,
            cache_dir="./.cache/embeddings",
            max_length=512
        )
        logger.info(f"Successfully initialized embedding model: {NOMIC_EMBED_MODEL}")
        return embed_model
    except Exception as e:
        logger.error(f"Embedding model initialization failed: {str(e)}")
        st.error(f"Failed to initialize embedding model: {str(e)}")
        return None
