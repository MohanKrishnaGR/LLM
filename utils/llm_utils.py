"""
Language Model utilities for initializing and managing LLM instances.
Provides caching and error handling for LLM and embedding model initialization.
"""

import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core.llms import LLM
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from config import EMBEDDING_MODEL_ID, LLAMA4_MODEL # Import EMBEDDING_MODEL_ID
import logging
from typing import Optional

logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Connecting to Embedding Model...")
def get_embedding_model() -> Optional[FastEmbedEmbedding]:
    """
    Initialize and cache the embedding model.
    
    Returns:
        FastEmbedEmbedding: Initialized embedding model instance.
        None: If initialization fails.
    """
    try:
        embed_model = FastEmbedEmbedding(model_name=EMBEDDING_MODEL_ID)
        logger.info(f"Successfully initialized embedding model: {EMBEDDING_MODEL_ID}")
        return embed_model
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {EMBEDDING_MODEL_ID}: {e}", exc_info=True)
        st.error(f"Error initializing embedding model ({EMBEDDING_MODEL_ID}): {e}")
        return None

@st.cache_resource(show_spinner="Connecting to LLM...")
def get_llm(model_name: str, api_key: Optional[str] = None) -> Optional[LLM]:
    """
    Initialize and cache the Language Model.
    
    Args:
        model_name: Name of the LLM model to initialize.
        api_key: Optional API key for model access.
        
    Returns:
        LLM: Initialized language model instance.
        None: If initialization fails.
    """
    if not api_key:
        if "groq_api_key_sidebar" in st.session_state and st.session_state["groq_api_key_sidebar"]:
            api_key = st.session_state["groq_api_key_sidebar"]
        else:
            logger.error("Groq API Key not found in session state or direct input.")
            st.error("Groq API Key is required. Please enter it in the sidebar.")
            return None
    
    try:
        llm = Groq(model=model_name, api_key=api_key)
        logger.info(f"Successfully initialized LLM: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM {model_name}: {e}", exc_info=True)
        st.error(f"Error initializing LLM ({model_name}): {e}")
        return None

def get_default_llm_for_judging(api_key: Optional[str] = None) -> Optional[LLM]:
    """
    Get the default LLM instance for judging tasks.
    
    Args:
        api_key: Optional API key for model access.
        
    Returns:
        LLM: Initialized language model instance for judging.
        None: If initialization fails.
    """
    from config import JUDGE_MODEL_NAME
    return get_llm(model_name=JUDGE_MODEL_NAME, api_key=api_key)