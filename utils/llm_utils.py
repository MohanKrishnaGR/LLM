# LLM/utils/llm_utils.py
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
    """Initializes and returns the embedding model using EMBEDDING_MODEL_ID from config."""
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
    """Initializes and returns the Language Model."""
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
    """Gets the default LLM specified for judging tasks."""
    # Assuming JUDGE_MODEL_NAME is defined in config, if not, fallback to a default like LLAMA4_MODEL
    from config import JUDGE_MODEL_NAME
    return get_llm(model_name=JUDGE_MODEL_NAME, api_key=api_key)