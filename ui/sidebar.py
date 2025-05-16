# LLM/ui/sidebar.py
# This file handles the rendering of the Streamlit sidebar for user configuration.

import streamlit as st
from config import (
    LLAMA4_MODEL, DEEPSEEK_MODEL, NOMIC_EMBED_MODEL, DATA_DIR,
    GRAPH_INDEX_DIR_PREFIX, INDEX_DIR_PREFIX
)
from utils.llm_utils import get_embedding_model, get_llm
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, KnowledgeGraphIndex, Settings,
    StorageContext
)
from llama_index.core.graph_stores import SimpleGraphStore
# from llama_index.retrievers.bm25 import BM25Retriever # If using BM25Retriever
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
    elif "groq_api_key_sidebar" in st.session_state:
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

    index_type_options = ["Vector Index", "Knowledge Graph Index"]
    st.session_state.selected_index_type = st.sidebar.selectbox(
        "Choose Index Type:",
        index_type_options,
        key="selected_index_type_key",
        help="Select the type of index to build and use for retrieval."
    )

    col1, col2 = st.sidebar.columns(2)

    if col1.button("Build Index", key="build_index_button"):
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            st.sidebar.error(f"No documents found in '{DATA_DIR}' directory!")
        elif "groq_api_key_sidebar" not in st.session_state or not st.session_state["groq_api_key_sidebar"]:
            st.sidebar.error("Groq API Key is required to build the index (for LLM-based operations like triplet extraction).")
        else:
            try:
                with st.spinner(f"Building {st.session_state.selected_index_type}... Please wait."):
                    logger.info(f"Starting to build {st.session_state.selected_index_type}.")
                    # Load documents for potential use in BM25 or other retrievers
                    # Storing documents in session state for potential use by BM25Retriever
                    # This is a simple way, for large datasets, consider a more robust approach.
                    if 'loaded_documents' not in st.session_state or not st.session_state.loaded_documents:
                        st.session_state.loaded_documents = SimpleDirectoryReader(DATA_DIR).load_data()
                    documents = st.session_state.loaded_documents
                    logger.info(f"Loaded/Retrieved {len(documents)} documents from {DATA_DIR}.")


                    embed_model = get_embedding_model()
                    if not embed_model:
                        st.sidebar.error("Failed to initialize embedding model.")
                        logger.error("Failed to initialize embedding model for indexing.")
                        return

                    llm_for_indexing = get_llm(
                        st.session_state.selected_model,
                        st.session_state.get("groq_api_key_sidebar")
                    )
                    if not llm_for_indexing:
                        st.sidebar.error("Failed to initialize LLM for indexing. Check API key and model selection.")
                        logger.error("Failed to initialize LLM for indexing.")
                        return

                    Settings.llm = llm_for_indexing
                    Settings.embed_model = embed_model
                    Settings.chunk_size = 512
                    Settings.chunk_overlap = 20
                    logger.info(f"LlamaIndex Settings configured: LLM={llm_for_indexing.__class__.__name__}, EmbedModel={embed_model.__class__.__name__}")

                    index = None
                    if st.session_state.selected_index_type == "Vector Index":
                        index = VectorStoreIndex.from_documents(
                            documents,
                            show_progress=True
                        )
                        st.session_state.vector_index = index
                        st.session_state.knowledge_graph_index = None
                        st.sidebar.success("Vector Index built successfully!")
                        logger.info("Vector Index built and stored in session state.")

                    elif st.session_state.selected_index_type == "Knowledge Graph Index":
                        graph_store = SimpleGraphStore()
                        storage_context = StorageContext.from_defaults(graph_store=graph_store)

                        index = KnowledgeGraphIndex.from_documents(
                            documents,
                            storage_context=storage_context,
                            max_triplets_per_chunk=5,
                            include_embeddings=True,
                            show_progress=True
                        )
                        st.session_state.knowledge_graph_index = index
                        st.session_state.vector_index = None
                        st.sidebar.success("Knowledge Graph Index built successfully!")
                        logger.info("Knowledge Graph Index built and stored in session state.")

            except Exception as e:
                logger.error(f"Error building index: {str(e)}", exc_info=True)
                st.sidebar.error(f"Error building index: {str(e)}")

    if col2.button("Clear Index", key="clear_index_button"):
        if "vector_index" in st.session_state:
            del st.session_state.vector_index
        if "knowledge_graph_index" in st.session_state:
            del st.session_state.knowledge_graph_index
        if "loaded_documents" in st.session_state:
            del st.session_state.loaded_documents # Clear loaded docs too
        st.sidebar.success("Current index and loaded documents cleared from session!")
        logger.info("Index and loaded documents cleared from session state.")

    # RAG Parameters
    st.sidebar.subheader("🔍 RAG Parameters")
    st.session_state.top_k = st.sidebar.slider(
        "Top-K Items to Retrieve (Dense)",
        min_value=1,
        max_value=10,
        value=3,
        key="top_k_retrieval_slider",
        help="Number of documents/contexts to retrieve via vector search."
    )
    
    # Hybrid Search Settings
    st.session_state.search_mode = st.sidebar.selectbox(
        "Search Mode:",
        ["Vector Only", "Hybrid (Vector + Sparse Fusion)"], # Add more modes like "BM25 Only" if implemented
        index=0,
        key="search_mode_key",
        help="Strategy for document retrieval. Hybrid mode combines dense and sparse retrieval."
    )
    if st.session_state.search_mode == "Hybrid (Vector + Sparse Fusion)":
        st.session_state.sparse_top_k = st.sidebar.slider(
            "Sparse Top-K (for Hybrid)",
            min_value=1,
            max_value=10,
            value=3,
            key="sparse_top_k_slider",
            help="Number of documents to retrieve via sparse (keyword-like) search in hybrid mode."
        )


    # Graph RAG specific parameters
    if st.session_state.get("selected_index_type") == "Knowledge Graph Index":
        st.session_state.graph_retriever_mode = st.sidebar.selectbox(
            "Graph Retriever Mode:",
            ["keyword", "embedding", "hybrid"],
            index=0,
            key="graph_retriever_mode_key",
            help="Method used by the Knowledge Graph retriever."
        )
        st.session_state.graph_traversal_depth = st.sidebar.slider(
            "Graph Traversal Depth", 1, 5, 2,
            key="graph_traversal_depth_slider",
            help="Depth for graph traversal in certain KG retrieval modes (e.g., ontology, or custom traversals)."
        )
        # Add other KG specific params like:
        # st.session_state.kg_embedding_similarity_cutoff = st.sidebar.slider(
        # "KG Embedding Similarity Cutoff", 0.0, 1.0, 0.7, 
        # help="Similarity cutoff for KG embedding based retrieval parts."
        # )


    # Advanced Features
    st.sidebar.subheader("✨ Advanced Features")

    # Query Transformation
    st.session_state.enable_query_transformation = st.sidebar.checkbox(
        "Enable Query Transformation",
        value=False,
        key="enable_query_transformation_checkbox",
        help="Use LLM to rephrase/expand the query before retrieval."
    )
    if st.session_state.enable_query_transformation:
        st.session_state.query_transformation_mode = st.sidebar.selectbox(
            "Query Transformation Mode:",
            ["Default Expansion", "Hypothetical Document (HyDE)"], # Add more modes
            index=0,
            key="query_transformation_mode_key",
            help="Method for query transformation."
        )


    st.session_state.enable_reranker = st.sidebar.checkbox(
        "Enable Re-ranker",
        value=True, # Default to True as it's generally useful
        key="enable_reranker_checkbox",
        help="Re-rank retrieved text documents for relevance using a cross-encoder model."
    )
    if st.session_state.enable_reranker:
        # Ensure rerank_top_n is sensible based on combined retrieval for hybrid
        max_rerank_input = st.session_state.top_k
        if st.session_state.search_mode == "Hybrid (Vector + Sparse Fusion)":
            max_rerank_input += st.session_state.get("sparse_top_k", 3)
        
        st.session_state.rerank_top_n = st.sidebar.slider(
            "Top-N After Re-ranking",
            min_value=1,
            max_value=max(1, max_rerank_input), # Cannot be more than total retrieved before rerank
            value=min(3, max(1, max_rerank_input)),
            key="rerank_top_n_slider",
            help="Number of documents to keep after re-ranking."
        )

    st.session_state.enable_llm_judge = st.sidebar.checkbox(
        "Enable LLM-as-a-Judge",
        value=False,
        key="enable_llm_judge_checkbox",
        help="Use an LLM to evaluate the generated response."
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Data Directory: `{DATA_DIR}`\n\n"
        "Add your `.txt`, `.md`, `.pdf` files to this directory before building the index."
    )   