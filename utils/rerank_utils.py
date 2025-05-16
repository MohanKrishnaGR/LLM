import streamlit as st
from config import RERANKER_MODEL_NAME
from llama_index.core.schema import NodeWithScore

try:
    from llama_index.postprocessor.sentence_transformers_rerank import SentenceTransformerRerank
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

def functional_reranker(nodes_with_scores: list[NodeWithScore], query: str, rerank_top_n: int = 3):
    if not RERANKER_AVAILABLE:
        st.warning("Re-ranker not installed.")
        return nodes_with_scores

    try:
        reranker = SentenceTransformerRerank(model=RERANKER_MODEL_NAME, top_n=rerank_top_n)
        return reranker.postprocess_nodes(nodes_with_scores, query_str=query)
    except Exception as e:
        st.error(f"Re-ranking failed: {e}")
        return nodes_with_scores
