# LLM/main.py
# Main Streamlit application script

import streamlit as st
from utils.directory_utils import ensure_data_directory # This is important
from ui.sidebar import render_sidebar
from ui.main_ui import handle_main_ui
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="Advanced RAG Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"
)

st.title("🔬 Advanced RAG Comparison and Analysis")

# Ensure the necessary data directory exists AT THE START
# This is crucial before any sidebar logic that might list files from it
ensure_data_directory()

render_sidebar()
handle_main_ui()

st.markdown("---")
st.markdown(
    "Developed by **Mohan Krishna G R** | Exploring AI/RAG? Let's "
    "[connect on LinkedIn](https://www.linkedin.com/in/grmk/) | "
    "[Portfolio](https://mohankrishnagr.github.io/) "
)