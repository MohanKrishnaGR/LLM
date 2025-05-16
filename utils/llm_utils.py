import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from config import NOMIC_EMBED_MODEL

def get_llm(model_name: str, api_key: str):
    if not api_key:
        return None
    try:
        return Groq(model=model_name, api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        return None

def get_embedding_model():
    try:
        return FastEmbedEmbedding(model_name=NOMIC_EMBED_MODEL)
    except Exception as e:
        st.error(f"Embedding model init failed: {e}")
        return None
