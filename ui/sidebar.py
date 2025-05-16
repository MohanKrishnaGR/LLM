import streamlit as st
from config import LLAMA4_MODEL, DEEPSEEK_MODEL, NOMIC_EMBED_MODEL, DATA_DIR

def render_sidebar():
    st.sidebar.header("Configuration")
    st.sidebar.subheader("API Keys")

    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", key="groq_api_key_sidebar_input")
    if groq_api_key:
        st.session_state["groq_api_key_sidebar"] = groq_api_key

    st.sidebar.subheader("LLM Selection")
    models = {
        "Llama3 70b (Meta)": LLAMA4_MODEL,
        "DeepSeek Coder 33b (DeepSeek)": DEEPSEEK_MODEL
    }
    selected = st.sidebar.selectbox("Choose LLM:", list(models.keys()), key="selected_model_key")
    st.session_state.selected_model = models[selected]

    st.sidebar.subheader("RAG Parameters")
    st.session_state.top_k = st.sidebar.slider("Top-K Documents to Retrieve", 1, 10, 3, key="top_k_retrieval_slider")

    st.sidebar.subheader("Advanced Features")
    st.session_state.enable_reranker = st.sidebar.checkbox("Enable Re-ranker", value=False)
    if st.session_state.enable_reranker:
        st.session_state.rerank_top_n = st.sidebar.slider("Top-N After Re-ranking", 1, st.session_state.top_k, 3)
    st.session_state.enable_llm_judge = st.sidebar.checkbox("Enable LLM-as-a-Judge", value=False)

    st.sidebar.markdown("---")
    st.sidebar.info(f"Data Directory: `{DATA_DIR}`\nAdd your .txt, .md, .pdf files before indexing.")
