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
from utils.directory_utils import (
    ensure_data_directory,
    save_uploaded_file,
    delete_file_from_data_dir,
    list_files_in_data_dir
)
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, KnowledgeGraphIndex, Settings,
    StorageContext, load_index_from_storage
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.schema import Document

from llama_index.readers.file import (
    PptxReader,
    DocxReader,
    PyMuPDFReader,
    MarkdownReader,
    FlatReader,
    PandasExcelReader
)
import os
import logging
import shutil

logger = logging.getLogger(__name__)

FILE_READER_MAP = {
    ".pdf": PyMuPDFReader(),
    ".docx": DocxReader(),
    ".pptx": PptxReader(),
    ".xlsx": PandasExcelReader(sheet_name=None), # Process all sheets
    ".txt": FlatReader(),
    ".md": MarkdownReader(),
}

def get_persist_dir(index_type_name: str) -> str:
    safe_name = index_type_name.replace(" ", "_").lower()
    return os.path.join(PERSIST_DIR_BASE, safe_name)

def load_or_initialize_index(selected_index_type: str, llm_for_indexing, embed_model):
    persist_dir = get_persist_dir(selected_index_type)
    logger.info(f"Attempting to load or initialize index of type '{selected_index_type}' from persist_dir: {persist_dir}")

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            logger.info(f"Attempting to load existing {selected_index_type} from {persist_dir}...")
            storage_context_args = {"persist_dir": persist_dir}

            if selected_index_type == "Knowledge Graph Index":
                graph_store_file_path = os.path.join(persist_dir, "graph_store.json")
                graph_store_dir_path = os.path.join(persist_dir, "graph_store")
                if os.path.exists(graph_store_file_path) or os.path.exists(graph_store_dir_path) :
                     logger.info(f"Potential KG graph_store data found in {persist_dir}.")
                else:
                    logger.warning(f"Knowledge Graph store data not explicitly found at {graph_store_file_path} or {graph_store_dir_path}. "
                                   f"The index at {persist_dir} might be incomplete. Will attempt to load normally.")

            storage_context = StorageContext.from_defaults(**storage_context_args)
            index = load_index_from_storage(storage_context)
            logger.info(f"Successfully loaded existing {selected_index_type} from {persist_dir}. Index object ID: {id(index)}")
            return index
        except Exception as e:
            logger.error(f"Failed to load index from {persist_dir}: {e}. Clearing directory and rebuilding if files exist.", exc_info=True)
            if os.path.exists(persist_dir):
                try:
                    shutil.rmtree(persist_dir)
                    logger.info(f"Cleared potentially corrupted/incomplete persistence directory: {persist_dir}")
                except Exception as rmtree_e:
                    logger.error(f"Error clearing directory {persist_dir}: {rmtree_e}")
            return None # Indicate failure to load

    logger.info(f"No persisted {selected_index_type} found at {persist_dir} or directory is empty. Index will be built from scratch if documents are available.")
    return None

def render_sidebar():
    st.sidebar.markdown("## 🛠️ RAG Configuration Workbench")
    st.sidebar.markdown("_Adjust settings to explore RAG techniques._")
    st.sidebar.markdown("---")

    ensure_data_directory()

    current_api_key = st.session_state.get("groq_api_key_sidebar", "")
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        value=current_api_key,
        type="password",
        key="groq_api_key_sidebar_input_v3", # Changed key to avoid conflict if old one persists
        help="Enter your Groq API key to use Groq-based LLMs."
    )
    if groq_api_key:
        st.session_state["groq_api_key_sidebar"] = groq_api_key


    st.sidebar.subheader("🧠 LLM Selection")
    models = {
        "Llama3 70b (Meta)": LLAMA4_MODEL,
        "DeepSeek-r1-distill-llama-70b": DEEPSEEK_MODEL
    }
    selected_model_display_name = st.sidebar.selectbox(
        "Choose LLM:",
        list(models.keys()),
        index=list(models.values()).index(st.session_state.get("selected_model", LLAMA4_MODEL)),
        key="selected_model_key",
        help="Select the Large Language Model to use for generation and potentially indexing."
    )
    st.session_state.selected_model = models[selected_model_display_name]

    st.sidebar.subheader("🗂️ Document Management")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents for RAG",
        type=[ext.lstrip('.') for ext in SUPPORTED_FILE_EXTENSIONS],
        accept_multiple_files=True,
        key="file_uploader_sidebar_widget"
    )

    if uploaded_files:
        success_count = 0
        for uploaded_file_obj in uploaded_files:
            if save_uploaded_file(uploaded_file_obj):
                success_count += 1
        if success_count > 0:
            st.sidebar.success(f"Successfully uploaded {success_count} file(s). Please 'Create / Update Index' to use them.")
            # st.session_state["file_uploader_sidebar_widget"] = [] # Clear uploader state
            st.rerun() # Rerun to reflect new files in directory listing if needed

    st.sidebar.markdown("##### Current Documents in RAG Data Directory:")
    files_in_data_dir = list_files_in_data_dir()
    if not files_in_data_dir:
        st.sidebar.caption(f"No supported files found in `{DATA_DIR}`. Use the uploader above.")
    else:
        for i, f_info in enumerate(files_in_data_dir):
            file_key_suffix = f_info['name'].replace(" ", "_").replace(".", "_") + f"_item_{i}"

            col_file, col_del_btn = st.sidebar.columns([0.8, 0.2])
            with col_file:
                st.caption(f"- {f_info['name']} ({f_info['size'] / 1024:.1f} KB)")
            with col_del_btn:
                if st.button("🗑️", key=f"del_btn_{file_key_suffix}", help=f"Delete {f_info['name']}"):
                    st.session_state[f"confirm_delete_for_{file_key_suffix}"] = True
                    st.rerun()

            if st.session_state.get(f"confirm_delete_for_{file_key_suffix}"):
                st.sidebar.warning(f"Are you sure you want to delete '{f_info['name']}'?")
                col_confirm_yes, col_confirm_no, _ = st.sidebar.columns([1,1,3])
                if col_confirm_yes.button("Yes", key=f"confirm_yes_del_{file_key_suffix}"):
                    if delete_file_from_data_dir(f_info['name']):
                        st.sidebar.success(f"Deleted '{f_info['name']}'. Please 'Create / Update Index'.")
                    else:
                        st.sidebar.error(f"Failed to delete '{f_info['name']}'.")
                    del st.session_state[f"confirm_delete_for_{file_key_suffix}"]
                    st.rerun()
                if col_confirm_no.button("No", key=f"confirm_no_del_{file_key_suffix}"):
                    del st.session_state[f"confirm_delete_for_{file_key_suffix}"]
                    st.rerun()
    st.sidebar.markdown("---")


    st.sidebar.subheader("📄 Document Processing")
    st.session_state.chunk_size_sidebar = st.sidebar.slider(
        "Chunk Size", 100, 2048, st.session_state.get("chunk_size_sidebar", DEFAULT_CHUNK_SIZE), 50,
        key="chunk_size_slider_sidebar", help="Size of text chunks for processing and embedding."
    )
    st.session_state.chunk_overlap_sidebar = st.sidebar.slider(
        "Chunk Overlap", 0, 512, st.session_state.get("chunk_overlap_sidebar", DEFAULT_CHUNK_OVERLAP), 10,
        key="chunk_overlap_slider_sidebar", help="Number of overlapping characters between chunks."
    )

    st.sidebar.subheader("📚 Index Management")
    index_type_options = ["Vector Index", "Knowledge Graph Index"]
    default_index_type = st.session_state.get("selected_index_type_key", index_type_options[0])
    if default_index_type not in index_type_options: default_index_type = index_type_options[0]

    st.session_state.selected_index_type = st.sidebar.selectbox(
        "Choose Index Type:",
        index_type_options,
        index=index_type_options.index(default_index_type),
        key="selected_index_type_key",
        help="Select the type of index to build and use."
    )

    col1_idx, col2_idx = st.sidebar.columns(2)

    if col1_idx.button("🔄 Create / Update Index", key="build_or_update_index_button"):
        if not os.path.isdir(DATA_DIR):
            st.sidebar.error(f"Data directory '{DATA_DIR}' does not exist!")
            logger.error(f"Data directory '{DATA_DIR}' not found during index build.")
            return

        if "groq_api_key_sidebar" not in st.session_state or not st.session_state["groq_api_key_sidebar"]:
            st.sidebar.error("Groq API Key is required for indexing. Please enter it above.")
            logger.warning("Groq API Key missing for indexing.")
            return

        try:
            with st.spinner(f"Preparing for {st.session_state.selected_index_type} creation/update..."):
                logger.info(f"Starting Create/Update for {st.session_state.selected_index_type}.")

                llm_for_indexing = get_llm(st.session_state.selected_model, st.session_state.get("groq_api_key_sidebar"))
                if not llm_for_indexing:
                    st.sidebar.error("Failed to initialize LLM for indexing. Check API key and model selection.")
                    return

                embed_model = get_embedding_model()
                if not embed_model:
                    st.sidebar.error("Failed to initialize embedding model for indexing.")
                    return

                Settings.llm = llm_for_indexing
                Settings.embed_model = embed_model
                Settings.chunk_size = st.session_state.chunk_size_sidebar
                Settings.chunk_overlap = st.session_state.chunk_overlap_sidebar
                logger.info(f"LlamaIndex Settings configured: LLM={st.session_state.selected_model}, Embeddings=default, ChunkSize={Settings.chunk_size}, ChunkOverlap={Settings.chunk_overlap}")

                index = load_or_initialize_index(st.session_state.selected_index_type, llm_for_indexing, embed_model)
                persist_dir = get_persist_dir(st.session_state.selected_index_type)
                current_doc_states_from_fs = scan_directory_for_document_states(DATA_DIR)
                st.session_state['current_doc_states_from_fs'] = current_doc_states_from_fs

                if not current_doc_states_from_fs and not index: # No files and no existing index
                    st.sidebar.warning(f"No supported documents in '{DATA_DIR}'. Upload documents to build an index.")
                    return

                tracked_doc_states = load_tracking_data(DOCUMENT_TRACKING_FILE)
                new_file_paths, modified_file_paths, deleted_doc_ids_from_tracking = compare_document_states(current_doc_states_from_fs, tracked_doc_states)

                logger.info(f"INITIAL FILE STATES: current_doc_states_from_fs keys: {list(current_doc_states_from_fs.keys())}")
                logger.info(f"INITIAL FILE STATES: tracked_doc_states keys: {list(tracked_doc_states.keys())}")
                logger.info(f"CHANGES DETECTED: New paths: {new_file_paths}, Modified paths: {modified_file_paths}, Deleted IDs: {deleted_doc_ids_from_tracking}")


            with st.spinner(f"Processing documents for {st.session_state.selected_index_type}..."):
                all_docs_for_current_index: list[Document] = []
                update_performed = False
                processed_new_files_names = []
                processed_modified_files_names = []
                processed_deleted_count = 0

                progress_message_area = st.sidebar.empty()

                if index: # Existing index
                    st.sidebar.info("Existing index found. Checking for updates...")
                    logger.info(f"Updating existing index. Index object ID before updates: {id(index)}")

                    if deleted_doc_ids_from_tracking:
                        logger.info(f"Deleting {len(deleted_doc_ids_from_tracking)} docs from index: {deleted_doc_ids_from_tracking}")
                        for doc_id in deleted_doc_ids_from_tracking:
                            file_name_for_log = tracked_doc_states.get(doc_id, {}).get('file_name', doc_id)
                            progress_message_area.info(f"Removing: {file_name_for_log}...")
                            try:
                                index.delete_ref_doc(doc_id, delete_from_docstore=True)
                                logger.info(f"Deleted doc_id '{doc_id}' (file: {file_name_for_log}) from index.")
                                update_performed = True
                                processed_deleted_count +=1
                            except Exception as e_del:
                                logger.error(f"Error deleting doc_id '{doc_id}' (file: {file_name_for_log}) from index: {e_del}", exc_info=True)
                        if processed_deleted_count > 0:
                            st.sidebar.write(f"Removed {processed_deleted_count} document(s) from the index.")

                    if modified_file_paths:
                        logger.info(f"Processing {len(modified_file_paths)} modified docs: {modified_file_paths}")
                        for i, file_path in enumerate(modified_file_paths):
                            file_name = current_doc_states_from_fs[file_path].get('file_name', os.path.basename(file_path))
                            progress_message_area.info(f"Updating: {file_name} ({i+1}/{len(modified_file_paths)})...")
                            logger.debug(f"Attempting to update file: {file_name} (path: {file_path})")
                            doc_id_to_update = file_path
                            try:
                                index.delete_ref_doc(doc_id_to_update, delete_from_docstore=True) # Delete old version first
                                logger.info(f"MODIFIED FILE: Deleted old ref_doc_id '{doc_id_to_update}' for file '{file_name}' before re-inserting.")
                            except Exception as e_del_mod:
                                logger.warning(f"MODIFIED FILE: Could not delete old ref_doc_id '{doc_id_to_update}' for '{file_name}' (may not exist or error): {e_del_mod}")

                            try:
                                modified_docs_loaded = SimpleDirectoryReader(input_files=[file_path], file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=False)
                                if modified_docs_loaded:
                                    logger.info(f"Loaded {len(modified_docs_loaded)} document objects from MODIFIED file: {file_name}")
                                    for doc_obj in modified_docs_loaded:
                                        if not doc_obj.text.strip():
                                            logger.warning(f"MODIFIED FILE: Document object from {file_name} (doc_id: {doc_obj.doc_id}, original_file_path: {doc_obj.metadata.get('file_path')}) has no text content. Skipping insertion.")
                                            continue
                                        logger.debug(f"MODIFIED FILE: Inserting doc_id: {doc_obj.doc_id} from file: {file_name} into index.")
                                        index.insert_ref_doc(doc_obj) # Re-insert new version
                                    update_performed = True
                                    processed_modified_files_names.append(file_name)
                                else:
                                    logger.warning(f"MODIFIED FILE: No document objects loaded from {file_name} using SimpleDirectoryReader.")
                            except Exception as e_mod_ins:
                                logger.error(f"Error processing modified file {file_name} for insertion: {e_mod_ins}", exc_info=True)
                                st.sidebar.warning(f"Error processing modified {file_name}. It might not be fully updated.")
                        if processed_modified_files_names:
                            st.sidebar.write(f"Updated {len(processed_modified_files_names)} document(s) in the index: {', '.join(processed_modified_files_names)}")

                    if new_file_paths:
                        logger.info(f"Processing {len(new_file_paths)} new docs: {new_file_paths}")
                        for i, file_path in enumerate(new_file_paths):
                            file_name = current_doc_states_from_fs[file_path].get('file_name', os.path.basename(file_path))
                            progress_message_area.info(f"Adding new: {file_name} ({i+1}/{len(new_file_paths)})...")
                            logger.debug(f"Attempting to add new file: {file_name} (path: {file_path})")
                            try:
                                new_docs_loaded = SimpleDirectoryReader(input_files=[file_path], file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=False)
                                if new_docs_loaded:
                                    logger.info(f"Loaded {len(new_docs_loaded)} document objects from NEW file: {file_name}")
                                    for doc_obj in new_docs_loaded:
                                        if not doc_obj.text.strip():
                                            logger.warning(f"NEW FILE: Document object from {file_name} (doc_id: {doc_obj.doc_id}, original_file_path: {doc_obj.metadata.get('file_path')}) has no text content. Skipping insertion.")
                                            continue
                                        logger.debug(f"NEW FILE: Inserting doc_id: {doc_obj.doc_id} from file: {file_name} into index.")
                                        index.insert_ref_doc(doc_obj)
                                    update_performed = True
                                    processed_new_files_names.append(file_name)
                                else:
                                    logger.warning(f"NEW FILE: No document objects loaded from {file_name} using SimpleDirectoryReader.")
                            except Exception as e_new_ins:
                                logger.error(f"Error processing new file {file_name} for insertion: {e_new_ins}", exc_info=True)
                                st.sidebar.warning(f"Error processing new {file_name}. It might not be added.")
                        if processed_new_files_names:
                            st.sidebar.write(f"Added {len(processed_new_files_names)} new document(s) to the index: {', '.join(processed_new_files_names)}")

                    logger.info(f"Index object ID after updates (before persist): {id(index)}")
                    progress_message_area.empty()
                    if not update_performed and not new_file_paths and not modified_file_paths and not deleted_doc_ids_from_tracking:
                        st.sidebar.info("No document changes detected. Index is up-to-date.")

                    # Consistently repopulate all_docs_for_current_index from current file system state
                    # This list is primarily for st.session_state.loaded_documents (UI context)
                    if current_doc_states_from_fs:
                        all_current_file_paths_for_docs = list(current_doc_states_from_fs.keys())
                        logger.info(f"Post-update: Re-loading all documents from {len(all_current_file_paths_for_docs)} current file paths for 'all_docs_for_current_index'.")
                        temp_docs_list = []
                        reloading_progress_area = st.sidebar.empty()
                        for i, f_path in enumerate(all_current_file_paths_for_docs):
                            fname = current_doc_states_from_fs[f_path].get('file_name', os.path.basename(f_path))
                            reloading_progress_area.caption(f"Verifying content: {fname} ({i+1}/{len(all_current_file_paths_for_docs)})...")
                            try:
                                docs = SimpleDirectoryReader(input_files=[f_path], file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=False)
                                temp_docs_list.extend(docs)
                            except Exception as e_reload:
                                logger.warning(f"Error reloading doc {fname} for final list: {e_reload}", exc_info=True)
                        all_docs_for_current_index = temp_docs_list
                        reloading_progress_area.empty()
                        logger.info(f"Post-update: 'all_docs_for_current_index' now contains {len(all_docs_for_current_index)} documents.")
                    else:
                        all_docs_for_current_index = []
                        logger.info("Post-update: 'current_doc_states_from_fs' is empty, so 'all_docs_for_current_index' is empty.")


                else: # Build from scratch
                    st.sidebar.info("No existing index or it was cleared. Building new index...")
                    logger.info("Building index from scratch.")
                    if not current_doc_states_from_fs:
                        st.sidebar.warning(f"No supported documents in '{DATA_DIR}' to build a new index. Please upload or add files.")
                    else:
                        all_initial_paths = list(current_doc_states_from_fs.keys())
                        logger.info(f"BUILD FROM SCRATCH: Loading {len(all_initial_paths)} docs from {DATA_DIR}: {all_initial_paths}")

                        loaded_initial_documents = []
                        for i, file_path in enumerate(all_initial_paths):
                            file_name = current_doc_states_from_fs[file_path].get('file_name', os.path.basename(file_path))
                            progress_message_area.info(f"Loading for new index: {file_name} ({i+1}/{len(all_initial_paths)})...")
                            logger.debug(f"BUILD FROM SCRATCH: Loading file {file_name} (path: {file_path})")
                            try:
                                docs_from_file = SimpleDirectoryReader(input_files=[file_path], file_extractor=FILE_READER_MAP, required_exts=SUPPORTED_FILE_EXTENSIONS).load_data(show_progress=False)
                                if docs_from_file:
                                    for doc_obj_build in docs_from_file:
                                        if not doc_obj_build.text.strip():
                                            logger.warning(f"BUILD FROM SCRATCH: Document object from {file_name} (doc_id: {doc_obj_build.doc_id}, original_file_path: {doc_obj_build.metadata.get('file_path')}) has no text content. Skipping.")
                                            continue
                                        loaded_initial_documents.append(doc_obj_build)
                                    logger.info(f"BUILD FROM SCRATCH: Loaded {len(docs_from_file)} valid document objects from {file_name}")
                                else:
                                     logger.warning(f"BUILD FROM SCRATCH: No document objects loaded from {file_name}")
                            except Exception as e_load_single:
                                logger.error(f"BUILD FROM SCRATCH: Error loading individual file {file_name}: {e_load_single}", exc_info=True)
                                st.sidebar.warning(f"Skipped {file_name} due to loading error.")

                        all_docs_for_current_index.extend(loaded_initial_documents)
                        progress_message_area.empty()

                        if not all_docs_for_current_index:
                            st.sidebar.error("No documents could be loaded from the files. Index building aborted. Check file contents/logs.")
                            return
                        else:
                            logger.info(f"BUILD FROM SCRATCH: Total {len(all_docs_for_current_index)} Document objects loaded. Building index...")
                            progress_message_area.info(f"All {len(all_docs_for_current_index)} documents loaded. Now building the {st.session_state.selected_index_type}...")

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
                            
                            if index:
                                logger.info(f"BUILD FROM SCRATCH: Index created successfully. Index object ID: {id(index)}")
                            else:
                                logger.error("BUILD FROM SCRATCH: Index object is None after attempting to build.")
                                st.sidebar.error(f"Failed to build {st.session_state.selected_index_type}. Check logs.")
                                return # Critical failure

                            progress_message_area.empty()
                            st.sidebar.write(f"Built new {st.session_state.selected_index_type} with {len(all_docs_for_current_index)} document(s).")
                            indexed_filenames = [doc.metadata.get('file_name', 'Unknown Filename') for doc in all_docs_for_current_index if hasattr(doc, 'metadata')]
                            if indexed_filenames:
                                with st.sidebar.expander("Documents included in the new index:", expanded=False):
                                    for name in indexed_filenames: st.caption(f"- {name}")
                            update_performed = True

                progress_message_area.empty()

                # Persist and update session state
                if index: # If an index exists (either loaded and updated, or newly built)
                    logger.info(f"Preparing to persist index. Index object ID: {id(index)}")
                    os.makedirs(persist_dir, exist_ok=True)
                    index.storage_context.persist(persist_dir=persist_dir)
                    logger.info(f"{st.session_state.selected_index_type} persisted to {persist_dir}. Index object ID after persist: {id(index)}")

                    save_tracking_data(DOCUMENT_TRACKING_FILE, current_doc_states_from_fs)
                    logger.info(f"Document tracking data saved. Current tracked file paths: {list(current_doc_states_from_fs.keys())}")

                    st.session_state.loaded_documents = all_docs_for_current_index
                    logger.info(f"Updated st.session_state.loaded_documents with {len(all_docs_for_current_index)} documents.")
                    logger.debug(f"Sample file names in st.session_state.loaded_documents: {[d.metadata.get('file_name', 'N/A') for d in all_docs_for_current_index[:3]]}")


                    if st.session_state.selected_index_type == "Vector Index":
                        st.session_state.vector_index = index
                        logger.info(f"Updated st.session_state.vector_index. New ID: {id(st.session_state.vector_index)}")
                        if "knowledge_graph_index" in st.session_state: del st.session_state.knowledge_graph_index
                    elif st.session_state.selected_index_type == "Knowledge Graph Index":
                        st.session_state.knowledge_graph_index = index
                        logger.info(f"Updated st.session_state.knowledge_graph_index. New ID: {id(st.session_state.knowledge_graph_index)}")
                        if "vector_index" in st.session_state: del st.session_state.vector_index
                    st.sidebar.success(f"{st.session_state.selected_index_type} is ready!")

                elif not current_doc_states_from_fs: # No files left, and index might be None or old
                    logger.info("No documents in data directory. Clearing index and tracking data.")
                    if os.path.exists(persist_dir):
                        try: shutil.rmtree(persist_dir); logger.info(f"Cleared persist_dir: {persist_dir}")
                        except Exception as e_rm_persist: logger.error(f"Error clearing persist_dir {persist_dir}: {e_rm_persist}")
                    if os.path.exists(DOCUMENT_TRACKING_FILE):
                        try: os.remove(DOCUMENT_TRACKING_FILE); logger.info(f"Cleared tracking file: {DOCUMENT_TRACKING_FILE}")
                        except OSError as e_clear_track: logger.error(f"Error clearing tracking file {DOCUMENT_TRACKING_FILE}: {e_clear_track}")

                    if "vector_index" in st.session_state: del st.session_state.vector_index
                    if "knowledge_graph_index" in st.session_state: del st.session_state.knowledge_graph_index
                    if "loaded_documents" in st.session_state: del st.session_state.loaded_documents
                    st.sidebar.info("All documents removed. Index has been cleared. Upload new documents to build an index.")

                elif not update_performed and (new_file_paths or modified_file_paths or deleted_doc_ids_from_tracking) and index:
                     st.sidebar.error("Index update attempted but no changes were applied or index state is unclear. Check logs.")
                     logger.error("Index update processed changes, but 'update_performed' is False with an existing index. This state might indicate an issue.")
                
                elif not update_performed and not (new_file_paths or modified_file_paths or deleted_doc_ids_from_tracking) and index :
                    st.sidebar.info("Index is up-to-date with the documents in the data directory.")
                    logger.info("Index loaded, no document changes needed. Index is up-to-date.")
                else: # Catch-all for unexpected scenarios
                    if current_doc_states_from_fs and not index:
                        st.sidebar.error("Index could not be created/updated despite documents being present. Check logs for errors during processing.")
                        logger.error("Index creation/update failed. 'index' is None but current_doc_states_from_fs is not empty and build was attempted.")
                    elif not index:
                        st.sidebar.warning("No index is active. Please add documents and create an index.")
                        logger.warning("Operation finished, but no index object is active.")


        except Exception as e:
            logger.error(f"CRITICAL ERROR during index process: {str(e)}", exc_info=True)
            st.sidebar.error(f"Critical error: {str(e)}. Check logs.")
            if 'progress_message_area' in locals() and hasattr(progress_message_area, 'empty'):
                progress_message_area.empty()

    if col2_idx.button("🧹 Clear Index & Cache", key="clear_index_button"):
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
                    cleared_msgs.append(f"{name} data")
                except Exception as e_clear:
                    logger.error(f"Error clearing {path}: {e_clear}")
        if os.path.exists(DOCUMENT_TRACKING_FILE):
            try:
                os.remove(DOCUMENT_TRACKING_FILE)
                cleared_msgs.append("Document tracking")
            except Exception as e_clear_track:
                logger.error(f"Error clearing tracking file {DOCUMENT_TRACKING_FILE}: {e_clear_track}")

        if cleared_msgs: st.sidebar.success(f"Cleared: {', '.join(cleared_msgs)}.")
        else: st.sidebar.info("No persisted data or tracking file found to clear.")
        logger.info("Index/cache cleared from session and persistence attempted.")
        st.rerun()

    st.sidebar.subheader("🔍 RAG Parameters")
    st.session_state.top_k = st.sidebar.slider("Top-K Dense", 1, 10, st.session_state.get("top_k", 3), key="top_k_slider_session")

    search_mode_options = ["Vector Only", "Hybrid (Vector + Sparse Fusion)"] # Add more as implemented
    current_search_mode = st.session_state.get("search_mode", "Vector Only")
    st.session_state.search_mode = st.sidebar.selectbox("Search Mode:", search_mode_options, index=search_mode_options.index(current_search_mode), key="search_mode_key_session")

    if st.session_state.search_mode == "Hybrid (Vector + Sparse Fusion)":
        st.session_state.sparse_top_k = st.sidebar.slider("Sparse Top-K", 1, 10, st.session_state.get("sparse_top_k", 3), key="sparse_top_k_key_session")


    if st.session_state.get("selected_index_type_key") == "Knowledge Graph Index":
        graph_retriever_options = ["keyword", "embedding", "hybrid"] # Ensure these match rag_workflow
        current_graph_mode = st.session_state.get("graph_retriever_mode", "keyword")
        st.session_state.graph_retriever_mode = st.sidebar.selectbox("Graph Retriever Mode:", graph_retriever_options, index=graph_retriever_options.index(current_graph_mode), key="graph_mode_key_session")
        st.session_state.graph_traversal_depth = st.sidebar.slider("Graph Traversal Depth", 1, 5, st.session_state.get("graph_traversal_depth", 2), key="graph_depth_key_session")
        st.session_state.kg_max_triplets = st.sidebar.slider("Max Triplets/Chunk (KG Build)", 1, 10, st.session_state.get("kg_max_triplets", 5), key="kg_triplets_key_session", help="Max triplets to extract per chunk during KG index build.")


    st.sidebar.subheader("✨ Advanced Features")
    st.session_state.enable_query_transformation = st.sidebar.checkbox("Enable Query Transformation", st.session_state.get("enable_query_transformation", False), key="query_transform_key_session")
    if st.session_state.enable_query_transformation:
        query_transform_options = ["Default Expansion", "Hypothetical Document (HyDE)"]
        current_query_transform_mode = st.session_state.get("query_transformation_mode", "Default Expansion")
        st.session_state.query_transformation_mode = st.sidebar.selectbox("Transform Mode:", query_transform_options, index=query_transform_options.index(current_query_transform_mode), key="query_transform_mode_key_session")

    st.session_state.enable_reranker = st.sidebar.checkbox("Enable Re-ranker", st.session_state.get("enable_reranker", True), key="reranker_key_session")
    if st.session_state.enable_reranker:
        # Ensure top_k is accessed correctly (from session_state if set, or a default)
        max_rerank_n = st.session_state.get("top_k", 3) # Use the actual top_k value
        default_rerank_top_n = min(st.session_state.get("rerank_top_n", 3), max(1, max_rerank_n)) # Ensure default is valid
        st.session_state.rerank_top_n = st.sidebar.slider(
            "Top-N After Re-ranking", 
            min_value=1, # Min value should always be 1
            max_value=max(1, max_rerank_n), # Max value is top_k (or 1 if top_k is less)
            value=default_rerank_top_n, 
            key="rerank_n_key_session", 
            help="Must be <= Top-K Dense."
        )
        # Ensure consistency after slider interaction
        if st.session_state.rerank_top_n > max_rerank_n :
            st.session_state.rerank_top_n = max_rerank_n
            st.rerun() # Rerun if value had to be corrected to update slider display

    st.session_state.enable_llm_judge = st.sidebar.checkbox("Enable LLM-as-a-Judge", st.session_state.get("enable_llm_judge", False), key="llm_judge_key_session")


    st.sidebar.markdown("---")
    st.sidebar.info(
        f"Data Dir: `{DATA_DIR}`\n"
        f"Supported: `{', '.join(SUPPORTED_FILE_EXTENSIONS)}`\n"
        f"Index Persist: `{PERSIST_DIR_BASE}`\n\n"
        "Upload/Add documents & 'Create / Update Index'."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the Developer")
    st.sidebar.markdown(
        "This application was developed by **Mohan Krishna G R**."
    )
    st.sidebar.markdown(
        "[View Portfolio](https://mohankrishnagr.github.io/) | "
        "[LinkedIn Profile](https://www.linkedin.com/in/grmk/)"
    )