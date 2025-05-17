"""
Re-ranking utilities for improving RAG response relevance.
Implements sentence transformer based re-ranking of retrieved nodes.
"""

import streamlit as st
from config import RERANKER_MODEL_ID
from llama_index.core.schema import NodeWithScore
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Attempt to import the reranker and set a flag
try:
    from llama_index.core.postprocessor import SentenceTransformerRerank
    SENTENCE_TRANSFORMER_RERANK_AVAILABLE = True
except ImportError:
    logger.warning(
        "SentenceTransformerRerank not found. "
        "Please install it via `pip install llama-index-postprocessor-sentence-transformer-rerank`. "
        "Re-ranking feature will be disabled."
    )
    SENTENCE_TRANSFORMER_RERANK_AVAILABLE = False
    SentenceTransformerRerank = None # Define it as None to avoid NameError later

@st.cache_resource(show_spinner="Initializing Re-ranker...")
def get_reranker(model_id: str, top_n: int) -> Optional[SentenceTransformerRerank]:
    """
    Initialize and cache the re-ranker model.
    
    Args:
        model_id: ID of the re-ranker model to use.
        top_n: Number of top results to return after re-ranking.
        
    Returns:
        SentenceTransformerRerank: Initialized re-ranker instance.
        None: If initialization fails.
    """
    if not SENTENCE_TRANSFORMER_RERANK_AVAILABLE:
        return None
    try:
        reranker = SentenceTransformerRerank(
            model=model_id,
            top_n=top_n
        )
        logger.info(f"Successfully initialized SentenceTransformerRerank with model: {model_id}, top_n: {top_n}")
        return reranker
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformerRerank with model {model_id}: {e}", exc_info=True)
        st.error(f"Error initializing re-ranker ({model_id}): {e}")
        return None

def functional_reranker(
    nodes_with_scores: List[NodeWithScore],
    query: str,
    rerank_top_n: int = 3
) -> List[NodeWithScore]:
    """
    Re-rank retrieved nodes based on relevance to the query.
    
    Args:
        nodes_with_scores: List of nodes with their initial relevance scores.
        query: User query for re-ranking context.
        rerank_top_n: Number of top results to return after re-ranking.
        
    Returns:
        List[NodeWithScore]: Re-ranked nodes with updated relevance scores.
    """
    if not SENTENCE_TRANSFORMER_RERANK_AVAILABLE:
        st.warning("Re-ranking is disabled because SentenceTransformerRerank is not available. Returning original nodes.")
        logger.warning("functional_reranker: Re-ranking skipped, component not available.")
        return nodes_with_scores

    if not nodes_with_scores:
        logger.info("functional_reranker: No nodes provided for re-ranking.")
        return []

    # Get the cached reranker instance
    # The top_n for the reranker instance itself is how many it *can* return.
    # The rerank_top_n here is how many we request from *this specific call*.
    # It's generally good if the instance's top_n is >= this function's rerank_top_n.
    # For simplicity, we use rerank_top_n for initializing the cached reranker,
    # assuming it matches the most common use.
    reranker_instance = get_reranker(model_id=RERANKER_MODEL_ID, top_n=rerank_top_n)

    if not reranker_instance:
        st.error("Re-ranker instance could not be initialized. Returning original nodes.")
        logger.error("functional_reranker: Reranker instance is None.")
        return nodes_with_scores

    try:
        logger.info(f"Re-ranking {len(nodes_with_scores)} nodes with query '{query}' using model {RERANKER_MODEL_ID} and top_n={rerank_top_n}.")
        # The reranker instance already has top_n set.
        # The postprocess_nodes method will use that instance's top_n.
        # If you want to dynamically change top_n per call without re-initializing,
        # you might need to adjust how get_reranker caches or is called.
        # For now, the reranker instance is configured with rerank_top_n from this function call.
        reranked_nodes = reranker_instance.postprocess_nodes(
            nodes_with_scores,
            query_str=query
            # No need to pass top_n again here if the instance is configured with it.
            # However, some postprocessors might accept it to override the instance's top_n.
            # SentenceTransformerRerank's postprocess_nodes uses the top_n from its initialization.
        )
        logger.info(f"Re-ranking complete. Returned {len(reranked_nodes)} nodes.")
        return reranked_nodes
    except Exception as e:
        logger.error(f"Re-ranking failed: {e}", exc_info=True)
        st.error(f"Re-ranking process failed: {e}")
        return nodes_with_scores # Fallback to original nodes on error
