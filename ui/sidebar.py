# LLM/ui/sidebar.py
# This file handles the rendering of the Streamlit sidebar for user configuration.

import streamlit as st
from config import (
    LLAMA4_MODEL, DEEPSEEK_MODEL, DATA_DIR, PERSIST_DIR_BASE, DOCUMENT_TRACKING_FILE
)
from utils.llm_utils import get_embedding_model, get_llm
from utils.document_tracker import (
    scan_directory_for_document_states,
    load_tracking_data,
    save_tracking_data,
    compare_document_states
)
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, KnowledgeGraphIndex, Settings,
    StorageContext, load_index_from_storage
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.schema import Document # Required for creating Document objects if needed

import os
import logging
import shutil # For clearing persisted directory

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_persist_dir(index_type_name: str) -> str:
    """Generates a persistence directory path for a given index type."""
    safe_name = index_type_name.replace(" ", "_").lower()
    return os.path.join(PERSIST_DIR_BASE, safe_name)

def load_or_initialize_index(selected_index_type: str, llm_for_indexing, embed_model):
    """
    Loads an index from persistence if available, otherwise returns None.
    Also configures LlamaIndex Settings.
    """
    Settings.llm = llm_for_indexing
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20
    logger.info(f"LlamaIndex Settings configured: LLM={llm_for_indexing.__class__.__name__}, EmbedModel={embed_model.__class__.__name__}")

    persist_dir = get_persist_dir(selected_index_type)
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            logger.info(f"Attempting to load existing {selected_index_type} from {persist_dir}...")
            storage_context_args = {"persist_dir": persist_dir}
            
            if selected_index_type == "Knowledge Graph Index":
                # For KG, graph_store needs to be explicitly passed if SimpleGraphStore was used.
                # Check if the graph_store specific file exists to confirm it was persisted.
                graph_store_persist_path = os.path.join(persist_dir, "graph_store")
                graph_store_file_path = os.path.join(graph_store_persist_path, "graph_store.json")
                if os.path.exists(graph_store_file_path):
                    logger.info(f"Found KG graph_store.json at {graph_store_file_path}, attempting to load.")
                    graph_store = SimpleGraphStore.from_persist_dir(persist_path=graph_store_persist_path)
                    storage_context_args["graph_store"] = graph_store
                else:
                    logger.warning(f"Knowledge Graph store file not found at {graph_store_file_path}. "
                                   f"The index at {persist_dir} might be incomplete or for a different type. "
                                   "Will attempt to load without explicit graph_store, but might fail or rebuild.")
            
            storage_context = StorageContext.from_defaults(**storage_context_args)
            index = load_index_from_storage(storage_context)
            logger.info(f"Successfully loaded existing {selected_index_type} from {persist_dir}.")
            return index
        except Exception as e:
            logger.error(f"Failed to load index from {persist_dir}: {e}. Clearing directory and rebuilding.", exc_info=True)
            if os.path.exists(persist_dir):
                try:
                    shutil.rmtree(persist_dir)
                    logger.info(f"Cleared potentially corrupted/incomplete persistence directory: {persist_dir}")
                except Exception as rmtree_e:
                    logger.error(f"Error clearing directory {persist_dir}: {rmtree_e}")
            return None
    logger.info(f"No persisted {selected_index_type} found at {persist_dir} or directory is empty. Index will be built from scratch.")
    return None


def render_sidebar():
    """
    Renders the sidebar UI components for configuration of API keys,
    LLM selection, index management, RAG parameters, and advanced features.
    """
    st.sidebar.header("⚙️ Configuration")

    st.sidebar.subheader("🔑 API Keys")
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        key="groq_api_key_sidebar_input",
        help="Enter your Groq API key to use Groq-based LLMs."
    )
    if groq_api_key:
        st.session_state["groq_api_key_sidebar"] = groq_api_key

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

    st.sidebar.subheader("📚 Index Management")

    index_type_options = ["Vector Index", "Knowledge Graph Index"]
    st.session_state.selected_index_type = st.sidebar.selectbox(
        "Choose Index Type:",
        index_type_options,
        key="selected_index_type_key",
        help="Select the type of index to build and use for retrieval."
    )

    col1, col2 = st.sidebar.columns(2)

    if col1.button("🔄 Create / Update Index", key="build_or_update_index_button"):
        if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR): # Check if dir and not empty
            st.sidebar.error(f"Data directory '{DATA_DIR}' is empty or does not exist! Add documents and try again.")
            return
        if "groq_api_key_sidebar" not in st.session_state or not st.session_state["groq_api_key_sidebar"]:
            st.sidebar.error("Groq API Key is required for indexing. Please enter it above.")
            return

        # This is the main try-except block for the entire indexing process
        try:
            with st.spinner(f"Processing documents and {st.session_state.selected_index_type}... Please wait."):
                logger.info(f"Starting to Create/Update {st.session_state.selected_index_type}.")
                
                llm_for_indexing = get_llm(
                    st.session_state.selected_model,
                    st.session_state.get("groq_api_key_sidebar")
                )
                if not llm_for_indexing:
                    st.sidebar.error("Failed to initialize LLM for indexing. Check API key and model selection.")
                    logger.error("Failed to initialize LLM for indexing.")
                    return # Exit if LLM setup fails

                embed_model = get_embedding_model()
                if not embed_model:
                    st.sidebar.error("Failed to initialize embedding model.")
                    logger.error("Failed to initialize embedding model for indexing.")
                    return # Exit if embed model setup fails
                
                index = load_or_initialize_index(st.session_state.selected_index_type, llm_for_indexing, embed_model)
                persist_dir = get_persist_dir(st.session_state.selected_index_type)

                current_doc_states_from_fs = scan_directory_for_document_states(DATA_DIR)
                tracked_doc_states = load_tracking_data(DOCUMENT_TRACKING_FILE)
                
                new_file_paths, modified_file_paths, deleted_doc_ids_from_tracking = \
                    compare_document_states(current_doc_states_from_fs, tracked_doc_states)

                all_docs_for_current_index: list[Document] = [] 

                if index: 
                    st.sidebar.info("Existing index found. Attempting incremental update...")
                    
                    if deleted_doc_ids_from_tracking:
                        logger.info(f"Deleting {len(deleted_doc_ids_from_tracking)} documents from index: {deleted_doc_ids_from_tracking}")
                        for doc_id in deleted_doc_ids_from_tracking:
                            try:
                                index.delete_ref_doc(doc_id, delete_from_docstore=True)
                                logger.info(f"Successfully deleted doc_id '{doc_id}' from index.")
                            except Exception as e_del:
                                logger.error(f"Error deleting doc_id '{doc_id}' from index: {e_del}", exc_info=True)
                        st.sidebar.write(f"Removed {len(deleted_doc_ids_from_tracking)} documents.")

                    if modified_file_paths:
                        logger.info(f"Updating {len(modified_file_paths)} modified documents in index: {modified_file_paths}")
                        for file_path in modified_file_paths:
                            doc_id_to_update = file_path 
                            try:
                                index.delete_ref_doc(doc_id_to_update, delete_from_docstore=True)
                                logger.info(f"Deleted old version of '{doc_id_to_update}' for update.")
                            except Exception as e_del_mod: 
                                logger.warning(f"Could not delete '{doc_id_to_update}' during update (might be benign if it was a new file misidentified, or already deleted): {e_del_mod}")

                            try:    
                                modified_doc_obj_list = SimpleDirectoryReader(input_files=[file_path]).load_data(show_progress=False)
                                if modified_doc_obj_list:
                                    for doc_obj in modified_doc_obj_list: # Iterate in case a file yields multiple documents
                                        index.insert(doc_obj) 
                                    # all_docs_for_current_index.extend(modified_doc_obj_list) # Will be reloaded later
                                    logger.info(f"Inserted updated version(s) from '{file_path}'.")
                                else:
                                    logger.warning(f"No document loaded from modified file: {file_path}")
                            except Exception as e_mod_ins:
                                logger.error(f"Error processing/inserting modified file {file_path}: {e_mod_ins}", exc_info=True)
                        st.sidebar.write(f"Updated {len(modified_file_paths)} documents.")

                    if new_file_paths:
                        logger.info(f"Adding {len(new_file_paths)} new documents to index: {new_file_paths}")
                        for file_path in new_file_paths:
                            try:
                                new_doc_obj_list = SimpleDirectoryReader(input_files=[file_path]).load_data(show_progress=False)
                                if new_doc_obj_list:
                                    for doc_obj in new_doc_obj_list:
                                        index.insert(doc_obj)
                                    # all_docs_for_current_index.extend(new_doc_obj_list) # Will be reloaded later
                                    logger.info(f"Inserted new document(s) from '{file_path}'.")
                                else:
                                    logger.warning(f"No document loaded from new file: {file_path}")
                            except Exception as e_new_ins:
                                logger.error(f"Error processing/inserting new file {file_path}: {e_new_ins}", exc_info=True)
                        st.sidebar.write(f"Added {len(new_file_paths)} new documents.")
                    
                    logger.info("Reloading all current documents from data directory for session state consistency after updates.")
                    if current_doc_states_from_fs:
                        all_docs_for_current_index = SimpleDirectoryReader(
                            input_files=list(current_doc_states_from_fs.keys())
                        ).load_data(show_progress=False) # Progress can be noisy here
                        logger.info(f"Reloaded {len(all_docs_for_current_index)} documents for session state.")
                    else: # Should not happen if updates occurred based on current_doc_states_from_fs, but as a safe guard.
                        all_docs_for_current_index = []
                        logger.info("Data directory is now empty after updates. Session documents cleared.")


                else: 
                    st.sidebar.info("No existing index found or forced rebuild. Building new index from scratch...")
                    if not current_doc_states_from_fs:
                        st.sidebar.warning("No documents found in data directory to build new index.")
                        # No explicit return here, will fall through to index being None
                    else:
                        logger.info(f"Loading all documents from {DATA_DIR} for initial build...")
                        all_initial_documents = SimpleDirectoryReader(
                            input_files=list(current_doc_states_from_fs.keys())
                        ).load_data(show_progress=True)
                        
                        all_docs_for_current_index.extend(all_initial_documents)

                        if not all_docs_for_current_index:
                            st.sidebar.error("No documents could be loaded. Index building aborted.")
                            # No explicit return here, will fall through
                        else:
                            if st.session_state.selected_index_type == "Vector Index":
                                index = VectorStoreIndex.from_documents(
                                    all_docs_for_current_index,
                                    show_progress=True
                                )
                            elif st.session_state.selected_index_type == "Knowledge Graph Index":
                                graph_store = SimpleGraphStore()
                                storage_context = StorageContext.from_defaults(graph_store=graph_store)
                                index = KnowledgeGraphIndex.from_documents(
                                    all_docs_for_current_index,
                                    storage_context=storage_context,
                                    max_triplets_per_chunk=st.session_state.get("kg_max_triplets", 5),
                                    include_embeddings=True,
                                    show_progress=True
                                )
                            st.sidebar.write(f"Built new index with {len(all_docs_for_current_index)} documents.")

                # Common final steps for both new build and update, only if index was created/loaded
                if index:
                    os.makedirs(persist_dir, exist_ok=True)
                    index.storage_context.persist(persist_dir=persist_dir)
                    logger.info(f"{st.session_state.selected_index_type} persisted to {persist_dir}")

                    save_tracking_data(DOCUMENT_TRACKING_FILE, current_doc_states_from_fs)
                    
                    st.session_state.loaded_documents = all_docs_for_current_index 
                    if st.session_state.selected_index_type == "Vector Index":
                        st.session_state.vector_index = index
                        if "knowledge_graph_index" in st.session_state: del st.session_state.knowledge_graph_index 
                    elif st.session_state.selected_index_type == "Knowledge Graph Index":
                        st.session_state.knowledge_graph_index = index
                        if "vector_index" in st.session_state: del st.session_state.vector_index 
                    
                    st.sidebar.success(f"{st.session_state.selected_index_type} is ready!")
                else:
                    st.sidebar.error("Index could not be created or updated. Check logs for details.")
            # End of 'with st.spinner' block
        # This is the 'except' for the main 'try' block of the button click
        except Exception as e: 
            logger.error(f"FATAL ERROR during index creation/update: {str(e)}", exc_info=True)
            st.sidebar.error(f"Critical error during indexing: {str(e)}. Check logs.")
        # End of 'if col1.button(...)' block

    if col2.button("🧹 Clear Index & Cache", key="clear_index_button"):
        if "vector_index" in st.session_state: del st.session_state.vector_index
        if "knowledge_graph_index" in st.session_state: del st.session_state.knowledge_graph_index
        if "loaded_documents" in st.session_state: del st.session_state.loaded_documents
        
        persist_dir_vector = get_persist_dir("Vector Index")
        persist_dir_kg = get_persist_dir("Knowledge Graph Index")
        
        cleared_paths = []
        for p_dir in [persist_dir_vector, persist_dir_kg]:
            if os.path.exists(p_dir):
                try:
                    shutil.rmtree(p_dir)
                    cleared_paths.append(p_dir)
                except Exception as e_clear:
                    logger.error(f"Error clearing directory {p_dir}: {e_clear}")
                    st.sidebar.warning(f"Could not fully clear {p_dir}.")
            
        if os.path.exists(DOCUMENT_TRACKING_FILE):
            try:
                os.remove(DOCUMENT_TRACKING_FILE)
                cleared_paths.append(DOCUMENT_TRACKING_FILE)
            except Exception as e_clear_track:
                logger.error(f"Error clearing tracking file {DOCUMENT_TRACKING_FILE}: {e_clear_track}")
                st.sidebar.warning(f"Could not clear {DOCUMENT_TRACKING_FILE}.")

        if cleared_paths:
            st.sidebar.success(f"Cleared data: {', '.join(cleared_paths)}")
        else:
            st.sidebar.info("No persisted index data found to clear.")
        logger.info("Index and loaded documents cleared from session and attempted from persistence.")

    st.sidebar.subheader("🔍 RAG Parameters")
    st.session_state.top_k = st.sidebar.slider(
        "Top-K Items to Retrieve (Dense)", 1, 10, 3, key="top_k_retrieval_slider"
    )
    st.session_state.search_mode = st.sidebar.selectbox(
        "Search Mode:", ["Vector Only", "Hybrid (Vector + Sparse Fusion)"], 0, key="search_mode_key"
    )
    if st.session_state.search_mode == "Hybrid (Vector + Sparse Fusion)":
        st.session_state.sparse_top_k = st.sidebar.slider(
            "Sparse Top-K (for Hybrid)", 1, 10, 3, key="sparse_top_k_slider"
        )

    if st.session_state.get("selected_index_type") == "Knowledge Graph Index":
        st.session_state.graph_retriever_mode = st.sidebar.selectbox(
            "Graph Retriever Mode:", ["keyword", "embedding", "hybrid"], 0, key="graph_retriever_mode_key"
        )
        st.session_state.graph_traversal_depth = st.sidebar.slider(
            "Graph Traversal Depth", 1, 5, 2, key="graph_traversal_depth_slider"
        )
        st.session_state.kg_max_triplets = st.sidebar.slider(
            "Max Triplets per Chunk (KG Build)", 1, 10, 5, key="kg_max_triplets_slider",
            help="Maximum triplets to extract per text chunk during KG index building."
        )

    st.sidebar.subheader("✨ Advanced Features")
    st.session_state.enable_query_transformation = st.sidebar.checkbox(
        "Enable Query Transformation", False, key="enable_query_transformation_checkbox"
    )
    if st.session_state.enable_query_transformation:
        st.session_state.query_transformation_mode = st.sidebar.selectbox(
            "Query Transformation Mode:", ["Default Expansion", "Hypothetical Document (HyDE)"], 0, key="query_transformation_mode_key"
        )
    st.session_state.enable_reranker = st.sidebar.checkbox(
        "Enable Re-ranker", True, key="enable_reranker_checkbox"
    )
    if st.session_state.enable_reranker:
        max_rerank_input = st.session_state.top_k
        if st.session_state.search_mode == "Hybrid (Vector + Sparse Fusion)" and "sparse_top_k" in st.session_state:
            max_rerank_input = max(max_rerank_input, st.session_state.sparse_top_k) # Max of the two if hybrid

        st.session_state.rerank_top_n = st.sidebar.slider(
            "Top-N After Re-ranking", 1, max(1, max_rerank_input), min(3, max(1, max_rerank_input)), key="rerank_top_n_slider"
        )
    st.session_state.enable_llm_judge = st.sidebar.checkbox(
        "Enable LLM-as-a-Judge", False, key="enable_llm_judge_checkbox"
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Data Directory: `{DATA_DIR}`\n"
        f"Index Persistence: `{PERSIST_DIR_BASE}`\n"
        "Add documents to data directory. Use 'Create / Update Index' to process."
    )