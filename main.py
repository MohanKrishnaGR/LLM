import streamlit as st
from utils.directory_utils import ensure_data_directory
from ui.sidebar import render_sidebar
from ui.main_ui import handle_main_ui

st.set_page_config(page_title="LLM Comparison", layout="wide")
st.title("🔬 Advanced LLM Comparison")

ensure_data_directory()
render_sidebar()
handle_main_ui()
