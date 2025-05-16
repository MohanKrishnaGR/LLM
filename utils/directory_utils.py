import os
import streamlit as st
from config import DATA_DIR

def ensure_data_directory():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        st.info(f"Created directory: {DATA_DIR}. Please add your documents here for RAG.")
    elif not os.listdir(DATA_DIR) and not st.session_state.get('vector_index'):
        st.warning(f"The directory '{DATA_DIR}' is empty. Please add documents.")
