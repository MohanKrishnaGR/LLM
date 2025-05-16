# LLM/main.py
# Main Streamlit application script

import streamlit as st
from utils.directory_utils import ensure_data_directory
from ui.sidebar import render_sidebar
from ui.main_ui import handle_main_ui # Corrected import: was run_async_main_ui

# Set page configuration for the Streamlit app
# This should be the first Streamlit command in your script, after imports.
st.set_page_config(
    page_title="Advanced RAG Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬" # Optional: Add a page icon
)

# Display the main title of the application
st.title("🔬 Advanced RAG Comparison and Analysis")

# Ensure the necessary data directory exists
ensure_data_directory()

# Render the sidebar for user configurations
render_sidebar()

# Handle the main UI elements and logic
handle_main_ui() # Corrected function call: was run_async_main_ui()

# Optional: Add a footer or other static content
st.markdown("---")
st.markdown(
    "Built with Streamlit and LlamaIndex. Explore advanced RAG techniques."
)

