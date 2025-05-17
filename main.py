"""
Main entry point for the RAG Comparison application.
Initializes the Streamlit interface and manages the application layout.
"""

import streamlit as st
from utils.directory_utils import ensure_data_directory
from ui.sidebar import render_sidebar
from ui.main_ui import handle_main_ui
from PIL import Image

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Advanced RAG Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=im
)

st.title("🔬 Advanced RAG Comparison and Analysis")

ensure_data_directory()

render_sidebar()
handle_main_ui()

st.markdown("---")
st.markdown(
    "Developed by **Mohan Krishna G R** | Exploring AI/RAG? Let's "
    "[connect on LinkedIn](https://www.linkedin.com/in/grmk/) | "
    "[Portfolio](https://mohankrishnagr.github.io/) "
)