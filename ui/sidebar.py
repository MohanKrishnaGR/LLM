# LLM/ui/sidebar.py
import streamlit as st
from config import (
    LLAMA4_MODEL, DEEPSEEK_MODEL, DATA_DIR, PERSIST_DIR_BASE, DOCUMENT_TRACKING_FILE,
    SUPPORTED_FILE_EXTENSIONS,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
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
from llama_index.core.schema import Document

# --- Added: Specific File Readers ---
from llama_index.readers.file import (
    PptxReader,
    DocxReader,
    PyMuPDFReader,      # For PDF
    MarkdownReader,
    FlatReader,         # For .txt
    PandasExcelReader   # For .xlsx (Corrected import location)
)
# --- End Added ---

import os
import logging
import shutil

logger = logging.getLogger(__name__)
# Ensure logging is configured if not done globally in main.py or streamlit_app.py
# For example: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- ADDED: File Reader Mapping ---
# This map defines which LlamaIndex reader to use for each supported file extension.
FILE_READER_MAP = {
    ".pdf": PyMuPDFReader(),
    ".docx": DocxReader(),
    ".pptx": PptxReader(),
    ".xlsx": PandasExcelReader(sheet_name=None),  # Reads all sheets from an Excel file
    ".txt": FlatReader(),                         # Simple text reader
    ".md": MarkdownReader(),
    # Add other extensions and their readers here if needed
}
# --- END ADDED ---


def get_persist_dir(index_type_name: str) -> str:
    """Generates a persistence directory path for a given index type."""
    safe_name = index_type_name.replace(" ", "_").lower()
    return os.path.join(PERSIST_DIR_BASE, safe_name)

def load_or_initialize_index(selected_index_type: str, llm_for_indexing, embed_model):
    """
    Loads an index from persistence if available, otherwise returns None.
    LlamaIndex Settings (LLM, embed_model, chunk_size, chunk_overlap) should be configured *before* calling this.
    """
    persist_dir = get_persist_dir(selected_index_type)
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir): # Check if directory exists and is not empty
        try:
            logger.info(f"Attempting to load existing {selected_index_type} from {persist_dir}...")
            storage_context_args = {"persist_dir": persist_dir}
            
            # Specific handling for KnowledgeGraphIndex if it uses SimpleGraphStore
            if selected_index_type == "Knowledge Graph Index":
                # Default path for SimpleGraphStore when persisted with KnowledgeGraphIndex
                # is typically within the main persist_dir, often as 'graph_store/graph_store.json'
                # However, load_index_from_storage handles this internally if SimpleGraphStore was used.
                # If a custom graph_store was passed during index creation AND it has its own persistence,
                # that would need special handling. Assuming default usage for now.
                # The key is that graph_store.json is inside the persist_dir for SimpleGraphStore.
                # We can check for its existence as an indicator.
                graph_store_file_path = os.path.join(persist_dir, "graph_store.json") # Path used by SimpleGraphStore's persist
                graph_store_dir_path = os.path.join(persist_dir, "graph_store") # Alternative path used by SimpleGraphStore.from_persist_dir

                if os.path.exists(graph_store_file_path) or os.path.exists(graph_store_dir_path) :
                     logger.info(f"Potential KG graph_store data found in {persist_dir}. Attempting to load with StorageContext.")
                     # No need to explicitly load SimpleGraphStore here if using default storage_context.persist
                     # load_index_from_storage should handle it.
                     # If it was persisted separately:
                     # if os.path.exists(graph_store_dir_path):
                     #    graph_store = SimpleGraphStore.from_persist_dir(persist_path=graph_store_dir_path)
                     #    storage_context_args["graph_store"] = graph_store

                else:
                    logger.warning(f"Knowledge Graph store data not explicitly found at {graph_store_file_path} or {graph_store_dir_path}. "
                                   f"The index at {persist_dir} might be incomplete or use a different graph store setup. "
                                   "Will attempt to load normally.")
            
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
            
    logger.info(f"No persisted {selected_index_type} found at {persist_dir} or directory is empty. Index will be built from scratch if documents are available.")
    return None


def render_sidebar():
    st.sidebar.markdown("## 🛠️ RAG Configuration Workbench") # Overall sidebar title
    st.sidebar.markdown("_Adjust settings to explore RAG techniques._")
    st.sidebar.markdown("---")

    # API Keys
    st.sidebar.subheader("🔑 API Keys")
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        key="groq_api_key_sidebar_input",
        help="Enter your Groq API key to use Groq-based LLMs."
    )
    if groq_api_key:
        st.session_state["groq_api_key_sidebar"] = groq_api_key

    # LLM Selection
    st.sidebar.subheader("🧠 LLM Selection")
    models = {
        "Llama3 70b (Meta)": LLAMA4_MODEL,
        "DeepSeek-r1-distill-llama-70b": DEEPSEEK_MODEL
    }
    selected_model_display_name = st.sidebar.selectbox(
        "Choose LLM:",
        list(models.keys()),
        key="selected_model_key",
        help="Select the Large Language Model to use for generation and potentially indexing."
    )
    st.session_state.selected_model = models[selected_model_display_name]

    st.sidebar.subheader("📄 Document Processing")
    st.session_state.chunk_size_sidebar = st.sidebar.slider(
        "Chunk Size", 100, 2048, st.session_state.get("chunk_size_sidebar", DEFAULT_CHUNK_SIZE), 50,
        key="chunk_size_slider_sidebar",
        help="Size of text chunks for processing and embedding."
    )
    st.session_state.chunk_overlap_sidebar = st.sidebar.slider(
        "Chunk Overlap", 0, 512, st.session_state.get("chunk_overlap_sidebar", DEFAULT_CHUNK_OVERLAP), 10,
        key="chunk_overlap_slider_sidebar",
        help="Number of overlapping characters between chunks."
    )

    st.sidebar.subheader("📚 Index Management")
    index_type_options = ["Vector Index", "Knowledge Graph Index"]
    if "selected_index_type_key" not in st.session_state:
        st.session_state.selected_index_type_key = index_type_options[0]

    st.session_state.selected_index_type = st.sidebar.selectbox(
        "Choose Index Type:",
        index_type_options,
        key="selected_index_type_key",
        help="Select the type of index to build and use."
    )

    col1, col2 = st.sidebar.columns(2)

    if col1.button("🔄 Create / Update Index", key="build_or_update_index_button"):
        if not os.path.isdir(DATA_DIR):
            st.sidebar.error(f"Data directory '{DATA_DIR}' does not exist! Create it and add documents.")
            logger.error(f"Data directory '{DATA_DIR}' not found.")
            return
        
        if "groq_api_key_sidebar" not in st.session_state or not st.session_state["groq_api_key_sidebar"]:
            st.sidebar.error("Groq API Key is required. Please enter it above.")
            logger.warning("Groq API Key missing.")
            return

        try:
            with st.spinner(f"Preparing for {st.session_state.selected_index_type}..."):
                logger.info(f"Starting Create/Update for {st.session_state.selected_index_type}.")
                
                llm_for_indexing = get_llm(st.session_state.selected_model, st.session_state.get("groq_api_key_sidebar"))
                if not llm_for_indexing:
                    st.sidebar.error("Failed to initialize LLM. Check API key and model.")
                    logger.error("LLM initialization failed for indexing.")
                    return

                embed_model = get_embedding_model()
                if not embed_model:
                    st.sidebar.error("Failed to initialize embedding model.")
                    logger.error("Embedding model initialization failed.")
                    return
                
                Settings.llm = llm_for_indexing
                Settings.embed_model = embed_model
                Settings.chunk_size = st.session_state.chunk_size_sidebar
                Settings.chunk_overlap = st.session_state.chunk_overlap_sidebar
                logger.info(f"Settings configured: LLM={Settings.llm.__class__.__name__}, Embed={Settings.embed_model.__class__.__name__}, ChunkSize={Settings.chunk_size}, Overlap={Settings.chunk_overlap}")

                index = load_or_initialize_index(st.session_state.selected_index_type, llm_for_indexing, embed_model)
                persist_dir = get_persist_dir(st.session_state.selected_index_type)

                current_doc_states_from_fs = scan_directory_for_document_states(DATA_DIR)
                
                if not current_doc_states_from_fs and not index:
                    st.sidebar.warning(f"No supported documents in '{DATA_DIR}'. Add documents (e.g., {', '.join(SUPPORTED_FILE_EXTENSIONS)}) and try again.")
                    logger.warning(f"No documents in {DATA_DIR} and no existing index.")
                    return

                tracked_doc_states = load_tracking_data(DOCUMENT_TRACKING_FILE)
                new_file_paths, modified_file_paths, deleted_doc_ids_from_tracking = compare_document_states(current_doc_states_from_fs, tracked_doc_states)

                all_docs_for_current_index: list[Document] = [] 
                update_performed = False

            with st.spinner(f"Processing documents for {st.session_state.selected_index_type}..."):
                if index:
                    st.sidebar.info("Existing index found. Checking for updates...")
                    
                    if deleted_doc_ids_from_tracking:
                        logger.info(f"Deleting {len(deleted_doc_ids_from_tracking)} docs: {deleted_doc_ids_from_tracking}")
                        for doc_id in deleted_doc_ids_from_tracking:
                            try:
                                index.delete_ref_doc(doc_id, delete_from_docstore=True)
                                logger.info(f"Deleted doc_id '{doc_id}' (file: {tracked_doc_states.get(doc_id, {}).get('file_name', doc_id)}).")
                                update_performed = True
                            except Exception as e_del:
                                logger.error(f"Error deleting doc_id '{doc_id}': {e_del}", exc_info=True)
                        st.sidebar.write(f"Removed {len(deleted_doc_ids_from_tracking)} documents.")

                    if modified_file_paths:
                        logger.info(f"Updating {len(modified_file_paths)} modified docs: {modified_file_paths}")
                        for file_path in modified_file_paths:
                            doc_id_to_update = file_path
                            try:
                                index.delete_ref_doc(doc_id_to_update, delete_from_docstore=True)
                                logger.info(f"Deleted old version of '{current_doc_states_from_fs[file_path].get('file_name', file_path)}' for update.")
                            except Exception as e_del_mod: 
                                logger.warning(f"Could not delete '{current_doc_states_from_fs[file_path].get('file_name', file_path)}' during update: {e_del_mod}")
                            
                            try:
                                modified_docs = SimpleDirectoryReader(input_files=[file_path], file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=False)
                                if modified_docs:
                                    for doc_obj in modified_docs: index.insert(doc_obj)
                                    logger.info(f"Inserted updated '{current_doc_states_from_fs[file_path].get('file_name', file_path)}'.")
                                    update_performed = True
                                else: logger.warning(f"No docs loaded from modified file: {current_doc_states_from_fs[file_path].get('file_name', file_path)}")
                            except Exception as e_mod_ins:
                                logger.error(f"Error inserting modified file {current_doc_states_from_fs[file_path].get('file_name', file_path)}: {e_mod_ins}", exc_info=True)
                        if update_performed or modified_file_paths: st.sidebar.write(f"Updated {len(modified_file_paths)} documents.")

                    if new_file_paths:
                        logger.info(f"Adding {len(new_file_paths)} new docs: {new_file_paths}")
                        for file_path in new_file_paths:
                            try:
                                new_docs = SimpleDirectoryReader(input_files=[file_path], file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=False)
                                if new_docs:
                                    for doc_obj in new_docs: index.insert(doc_obj)
                                    logger.info(f"Inserted new '{current_doc_states_from_fs[file_path].get('file_name', file_path)}'.")
                                    update_performed = True
                                else: logger.warning(f"No docs loaded from new file: {current_doc_states_from_fs[file_path].get('file_name', file_path)}")
                            except Exception as e_new_ins:
                                logger.error(f"Error inserting new file {current_doc_states_from_fs[file_path].get('file_name', file_path)}: {e_new_ins}", exc_info=True)
                        st.sidebar.write(f"Added {len(new_file_paths)} new documents.")
                    
                    if not update_performed and not new_file_paths and not modified_file_paths and not deleted_doc_ids_from_tracking:
                        st.sidebar.info("No document changes. Index is up-to-date.")
                    
                    if current_doc_states_from_fs:
                        all_paths = list(current_doc_states_from_fs.keys())
                        if all_paths:
                            logger.info(f"Reloading {len(all_paths)} docs for session state...")
                            all_docs_for_current_index = SimpleDirectoryReader(input_files=all_paths, file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=False)
                            logger.info(f"Reloaded {len(all_docs_for_current_index)} objects for session after updates.")
                        else: all_docs_for_current_index = []
                    else: all_docs_for_current_index = []

                else: # Build from scratch
                    st.sidebar.info("No existing index or it was cleared. Building new index...")
                    if not current_doc_states_from_fs:
                        st.sidebar.warning(f"No supported documents in '{DATA_DIR}' to build new index.")
                    else:
                        all_initial_paths = list(current_doc_states_from_fs.keys())
                        logger.info(f"Loading {len(all_initial_paths)} docs from {DATA_DIR} for initial build...")
                        
                        all_initial_documents = SimpleDirectoryReader(input_files=all_initial_paths, file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=True)
                        all_docs_for_current_index.extend(all_initial_documents)

                        if not all_docs_for_current_index:
                            st.sidebar.error("No documents loaded. Index building aborted. Check file contents/logs.")
                            logger.error(f"Failed to load Document objects from {len(all_initial_paths)} files.")
                        else:
                            logger.info(f"Loaded {len(all_docs_for_current_index)} Document objects. Building index...")
                            if st.session_state.selected_index_type == "Vector Index":
                                index = VectorStoreIndex.from_documents(all_docs_for_current_index, show_progress=True)
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
                            st.sidebar.write(f"Built new {st.session_state.selected_index_type} with {len(all_docs_for_current_index)} documents.")
                            update_performed = True

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

                elif not update_performed and (new_file_paths or modified_file_paths or deleted_doc_ids_from_tracking):
                     st.sidebar.error("Index update attempted but resulted in no valid index. Check logs.")
                     logger.error("Index update attempted, but 'index' is None. update_performed was False.")
                elif not current_doc_states_from_fs and not index : pass
                elif not update_performed and not (new_file_paths or modified_file_paths or deleted_doc_ids_from_tracking) and index : pass
                else:
                    st.sidebar.error("Index could not be created/updated. Check logs.")
                    logger.error("'index' is None, and other specific conditions weren't met.")

        except Exception as e: 
            logger.error(f"CRITICAL ERROR during index process: {str(e)}", exc_info=True)
            st.sidebar.error(f"Critical error: {str(e)}. Check logs.")

    if col2.button("🧹 Clear Index & Cache", key="clear_index_button"):
        if "vector_index" in st.session_state: del st.session_state.vector_index
        if "knowledge_graph_index" in st.session_state: del st.session_state.knowledge_graph_index
        if "loaded_documents" in st.session_state: del st.session_state.loaded_documents
        
        persist_dir_vector = get_persist_dir("Vector Index")
        persist_dir_kg = get_persist_dir("Knowledge Graph Index")
        
        cleared_msgs = []
        for name, path in [("Vector Index", persist_dir_vector), ("KG Index", persist_dir_kg)]:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    logger.info(f"Cleared persisted {name} directory: {path}")
                    cleared_msgs.append(f"{name} data")
                except Exception as e_clear:
                    logger.error(f"Error clearing {path}: {e_clear}")
                    st.sidebar.warning(f"Could not fully clear {name} from {path}.")
            
        if os.path.exists(DOCUMENT_TRACKING_FILE):
            try:
                os.remove(DOCUMENT_TRACKING_FILE)
                logger.info(f"Cleared tracking file: {DOCUMENT_TRACKING_FILE}")
                cleared_msgs.append("Document tracking")
            except Exception as e_clear_track:
                logger.error(f"Error clearing tracking file {DOCUMENT_TRACKING_FILE}: {e_clear_track}")
                st.sidebar.warning(f"Could not clear {DOCUMENT_TRACKING_FILE}.")

        if cleared_msgs: st.sidebar.success(f"Cleared: {', '.join(cleared_msgs)}.")
        else: st.sidebar.info("No persisted data or tracking file found to clear.")
        logger.info("Index/cache cleared from session and persistence attempted.")

    st.sidebar.subheader("🔍 RAG Parameters")
    st.session_state.top_k = st.sidebar.slider("Top-K Dense", 1, 10, 3, key="top_k_slider")
    st.session_state.search_mode = st.sidebar.selectbox(
        "Search Mode:", ["Vector Only", "Hybrid (Vector + Sparse Fusion)"], 0, key="search_mode_key"
    )
    if st.session_state.search_mode == "Hybrid (Vector + Sparse Fusion)":
        st.session_state.sparse_top_k = st.sidebar.slider("Sparse Top-K", 1, 10, 3, key="sparse_top_k_key")

    if st.session_state.get("selected_index_type_key") == "Knowledge Graph Index":
        st.session_state.graph_retriever_mode = st.sidebar.selectbox(
            "Graph Retriever Mode:", ["keyword", "embedding", "hybrid"], 0, key="graph_mode_key"
        )
        st.session_state.graph_traversal_depth = st.sidebar.slider("Graph Traversal Depth", 1, 5, 2, key="graph_depth_key")
        st.session_state.kg_max_triplets = st.sidebar.slider(
            "Max Triplets/Chunk (KG Build)", 1, 10, 5, key="kg_triplets_key",
            help="Max triplets to extract per chunk during KG index build."
        )

    st.sidebar.subheader("✨ Advanced Features")
    st.session_state.enable_query_transformation = st.sidebar.checkbox("Enable Query Transformation", False, key="query_transform_key")
    if st.session_state.enable_query_transformation:
        st.session_state.query_transformation_mode = st.sidebar.selectbox(
            "Transform Mode:", ["Default Expansion", "Hypothetical Document (HyDE)"], 0, key="query_transform_mode_key"
        )
    
    st.session_state.enable_reranker = st.sidebar.checkbox("Enable Re-ranker", True, key="reranker_key")
    if st.session_state.enable_reranker:
        max_rerank_n = st.session_state.top_k
        st.session_state.rerank_top_n = st.sidebar.slider(
            "Top-N After Re-ranking", 1, max(1, max_rerank_n), min(3, max(1, max_rerank_n)),
            key="rerank_n_key", help="Must be <= Top-K Dense."
        )
        if st.session_state.rerank_top_n > st.session_state.top_k:
            st.session_state.rerank_top_n = st.session_state.top_k
            
    st.session_state.enable_llm_judge = st.sidebar.checkbox("Enable LLM-as-a-Judge", False, key="llm_judge_key")

    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Data Dir: `{DATA_DIR}`\n"
        f"Supported: `{', '.join(SUPPORTED_FILE_EXTENSIONS)}`\n"
        f"Index Persist: `{PERSIST_DIR_BASE}`\n\n"
        "Add documents to data dir & 'Create / Update Index'."
    )
    st.sidebar.markdown("---") # Add a separator
    st.sidebar.markdown("### About the Developer")
    st.sidebar.markdown(
        "This application was developed by **Mohan Krishna G R**."
    )
    st.sidebar.markdown(
        "[View Portfolio](https://mohankrishnagr.github.io/) | "
        "[LinkedIn Profile](https://www.linkedin.com/in/grmk/)" 
    )
