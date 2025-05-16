# LLM/ui/sidebar.py
# This file handles the rendering of the Streamlit sidebar for user configuration.

import streamlit as st
from config import (
    LLAMA4_MODEL, DEEPSEEK_MODEL, NOMIC_EMBED_MODEL, DATA_DIR, 
    GRAPH_INDEX_DIR_PREFIX, INDEX_DIR_PREFIX # Ensure these are imported from config
)
from utils.llm_utils import get_embedding_model, get_llm
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, KnowledgeGraphIndex, Settings,
    StorageContext # ServiceContext is deprecated, use Settings and StorageContext
)
from llama_index.core.graph_stores import SimpleGraphStore # For in-memory graph store
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def render_sidebar():
    """
    Renders the sidebar UI components for configuration of API keys,
    LLM selection, index management, RAG parameters, and advanced features.
    """
    st.sidebar.header("⚙️ Configuration")
    
    # API Key Configuration
    st.sidebar.subheader("🔑 API Keys")
    groq_api_key = st.sidebar.text_input(
        "Groq API Key", 
        type="password", 
        key="groq_api_key_sidebar_input",
        help="Enter your Groq API key to use Groq-based LLMs."
    )
    if groq_api_key:
        st.session_state["groq_api_key_sidebar"] = groq_api_key
    elif "groq_api_key_sidebar" in st.session_state: # Clear if input is empty
        del st.session_state["groq_api_key_sidebar"]

    # LLM Selection
    st.sidebar.subheader("🧠 LLM Selection")
    models = {
        "Llama3 70b (Meta)": LLAMA4_MODEL,
        "DeepSeek Coder 33b (DeepSeek)": DEEPSEEK_MODEL
    }
    selected_model_display_name = st.sidebar.selectbox(
        "Choose LLM:", 
        list(models.keys()), 
        key="selected_model_key",
        help="Select the Large Language Model to use for generation."
    )
    st.session_state.selected_model = models[selected_model_display_name]

    # Index Management
    st.sidebar.subheader("📚 Index Management")

    # Index type selection
    index_type_options = ["Vector Index", "Knowledge Graph Index"]
    st.session_state.selected_index_type = st.sidebar.selectbox(
        "Choose Index Type:",
        index_type_options,
        key="selected_index_type_key",
        help="Select the type of index to build and use for retrieval."
    )

    col1, col2 = st.sidebar.columns(2)

    # Build Index Button
    if col1.button("Build Index", key="build_index_button"):
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            st.sidebar.error(f"No documents found in '{DATA_DIR}' directory!")
        elif "groq_api_key_sidebar" not in st.session_state or not st.session_state["groq_api_key_sidebar"]:
            st.sidebar.error("Groq API Key is required to build the index (for LLM-based operations like triplet extraction).")
        else:
            try:
                with st.spinner(f"Building {st.session_state.selected_index_type}... Please wait."):
                    logger.info(f"Starting to build {st.session_state.selected_index_type}.")
                    documents = SimpleDirectoryReader(DATA_DIR).load_data()
                    logger.info(f"Loaded {len(documents)} documents from {DATA_DIR}.")

                    embed_model = get_embedding_model()
                    if not embed_model:
                        st.sidebar.error("Failed to initialize embedding model.")
                        logger.error("Failed to initialize embedding model for indexing.")
                        return # Exit if embed_model fails

                    llm_for_indexing = get_llm(
                        st.session_state.selected_model,
                        st.session_state.get("groq_api_key_sidebar")
                    )
                    if not llm_for_indexing:
                        st.sidebar.error("Failed to initialize LLM for indexing. Check API key and model selection.")
                        logger.error("Failed to initialize LLM for indexing.")
                        return # Exit if LLM fails

                    # Configure LlamaIndex global settings
                    Settings.llm = llm_for_indexing
                    Settings.embed_model = embed_model
                    Settings.chunk_size = 512 # Example: set chunk size globally
                    Settings.chunk_overlap = 20 # Example: set chunk overlap globally
                    logger.info(f"LlamaIndex Settings configured: LLM={llm_for_indexing.__class__.__name__}, EmbedModel={embed_model.__class__.__name__}")

                    index = None
                    if st.session_state.selected_index_type == "Vector Index":
                        index = VectorStoreIndex.from_documents(
                            documents,
                            show_progress=True
                            # service_context is deprecated, Settings are used globally
                        )
                        st.session_state.vector_index = index
                        st.session_state.knowledge_graph_index = None # Clear other index type
                        st.sidebar.success("Vector Index built successfully!")
                        logger.info("Vector Index built and stored in session state.")

                    elif st.session_state.selected_index_type == "Knowledge Graph Index":
                        graph_store = SimpleGraphStore()
                        storage_context = StorageContext.from_defaults(graph_store=graph_store)
                        
                        index = KnowledgeGraphIndex.from_documents(
                            documents,
                            storage_context=storage_context,
                            max_triplets_per_chunk=5, # Number of knowledge triplets to extract per chunk
                            include_embeddings=True, # Generate embeddings for graph nodes for hybrid search
                            show_progress=True
                        )
                        st.session_state.knowledge_graph_index = index
                        st.session_state.vector_index = None # Clear other index type
                        st.sidebar.success("Knowledge Graph Index built successfully!")
                        logger.info("Knowledge Graph Index built and stored in session state.")
                    
                    # Optional: Persist the index to disk
                    # persist_dir_name = f"{INDEX_DIR_PREFIX if st.session_state.selected_index_type == 'Vector Index' else GRAPH_INDEX_DIR_PREFIX}{st.session_state.selected_model.replace('/', '_')}"
                    # if index and hasattr(index, 'storage_context'):
                    #     index.storage_context.persist(persist_dir=persist_dir_name)
                    #     st.sidebar.info(f"Index persisted to ./{persist_dir_name}")
                    #     logger.info(f"Index persisted to ./{persist_dir_name}")


            except Exception as e:
                logger.error(f"Error building index: {str(e)}", exc_info=True)
                st.sidebar.error(f"Error building index: {str(e)}")

    # Clear Index Button
    if col2.button("Clear Index", key="clear_index_button"):
        if "vector_index" in st.session_state:
            del st.session_state.vector_index
        if "knowledge_graph_index" in st.session_state:
            del st.session_state.knowledge_graph_index
        st.sidebar.success("Current index cleared from session!")
        logger.info("Index cleared from session state.")
        # Optionally, add logic here to clear persisted index directories if you implement persistence.

    # RAG Parameters
    st.sidebar.subheader("🔍 RAG Parameters")
    st.session_state.top_k = st.sidebar.slider(
        "Top-K Items to Retrieve", 
        min_value=1, 
        max_value=10, 
        value=3, 
        key="top_k_retrieval_slider",
        help="Number of documents or graph contexts to retrieve for the LLM."
    )

    # Graph RAG specific parameters
    if st.session_state.get("selected_index_type") == "Knowledge Graph Index":
        st.session_state.graph_retriever_mode = st.sidebar.selectbox(
            "Graph Retriever Mode:",
            ["keyword", "embedding", "hybrid"], # LlamaIndex supports these modes for KG retrievers
            index=0, # Default to keyword based retrieval
            key="graph_retriever_mode_key",
            help="Method used by the Knowledge Graph retriever. 'Hybrid' may require specific graph store support."
        )
        # Example for another graph-specific parameter
        # st.session_state.graph_traversal_depth = st.sidebar.slider(
        # "Graph Traversal Depth", 1, 5, 2, 
        # help="Depth for graph traversal in certain KG retrieval modes."
        # )

    # Advanced Features
    st.sidebar.subheader("✨ Advanced Features")
    st.session_state.enable_reranker = st.sidebar.checkbox(
        "Enable Re-ranker", 
        value=False, 
        key="enable_reranker_checkbox",
        help="Re-rank retrieved text documents for relevance using a cross-encoder model."
    )
    if st.session_state.enable_reranker:
        st.session_state.rerank_top_n = st.sidebar.slider(
            "Top-N After Re-ranking", 
            min_value=1, 
            max_value=st.session_state.top_k, # Cannot be more than top_k
            value=min(3, st.session_state.top_k), # Default to 3 or top_k if smaller
            key="rerank_top_n_slider",
            help="Number of documents to keep after re-ranking."
        )
    
    st.session_state.enable_llm_judge = st.sidebar.checkbox(
        "Enable LLM-as-a-Judge", 
        value=False, 
        key="enable_llm_judge_checkbox",
        help="Use an LLM to evaluate the generated response."
    )

    # Information Section
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Data Directory: `{DATA_DIR}`\n\n"
        "Add your `.txt`, `.md`, `.pdf` files to this directory before building the index."
    )
