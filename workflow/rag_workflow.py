# LLM/workflow/rag_workflow.py
# Defines the RAG (Retrieval Augmented Generation) workflow using LlamaIndex.

import streamlit as st
from typing import List, Optional, Dict, Any
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine # Can be useful for KG
from llama_index.core import (
    KnowledgeGraphIndex, 
    VectorStoreIndex, 
    Settings, 
    PromptTemplate # For custom prompts
)
from llama_index.core.response_synthesizers import get_response_synthesizer # Replaces CompactAndRefine directly
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.core.base.response.schema import Response # Not directly used, Response object is returned by synthesizer
from utils.rerank_utils import functional_reranker
from utils.judge_utils import functional_llm_as_judge
import logging
import json # For metadata display if needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrievalResult(Event):
    """
    Event to store the results from the retrieval step.
    """
    nodes: List[NodeWithScore]  # List of retrieved nodes with their scores
    original_query: str         # The original user query
    metadata: Dict[str, Any]    # Metadata about the retrieval process

class RAGWorkflow(Workflow):
    """
    Advanced RAG Workflow supporting both VectorStoreIndex and KnowledgeGraphIndex.
    It includes steps for retrieval, optional re-ranking, and response synthesis.
    """
    
    def __init__(self, llm: Any, index: Any, **kwargs):
        """
        Initializes the RAG workflow.
        Args:
            llm: The language model instance to use for generation.
            index: The LlamaIndex index instance (VectorStoreIndex or KnowledgeGraphIndex).
            **kwargs: Additional arguments for the base Workflow class.
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.index = index  # This can be VectorStoreIndex or KnowledgeGraphIndex
        
        # These are defaults, can be overridden or dynamically set
        self.similarity_cutoff = 0.7  # Default similarity threshold for vector search postprocessing
        self.max_tokens_for_llm = 4096 # Max tokens for LLM, if LLM itself is not configured

        # Ensure LlamaIndex global settings are updated with the provided LLM
        # Embed model should be set globally during index creation or app startup
        if Settings.llm is None or Settings.llm != self.llm:
            Settings.llm = self.llm
            logger.info(f"RAGWorkflow: Settings.llm updated to {self.llm.__class__.__name__}")

    def _process_text(self, text: Any) -> str:
        """
        Helper function to process text, ensuring it's a UTF-8 string and stripped.
        Args:
            text: The input text (can be of any type that can be cast to string).
        Returns:
            A processed string, or an empty string if processing fails.
        """
        if not isinstance(text, str):
            text = str(text)
        try:
            # Handle potential encoding issues and strip whitespace
            text = text.encode('utf-8', 'ignore').decode('utf-8')
            return text.strip()
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}", exc_info=True)
            return ""

    def _prepare_query(self, query: str) -> QueryBundle:
        """
        Prepares and validates the query by processing it.
        Args:
            query: The raw user query string.
        Returns:
            A QueryBundle object.
        Raises:
            ValueError: If the query is empty after processing.
        """
        processed_query = self._process_text(query)
        if not processed_query:
            logger.error("Query is empty after processing.")
            raise ValueError("Query is empty after processing")
        return QueryBundle(query_str=processed_query)

    @step
    async def retrieve(self, ev: StartEvent, query: str, top_k: int = 3, rerank_top_n: int = 3) -> RetrievalResult:
        """
        Retrieval step: Fetches relevant documents or graph contexts based on the index type.
        Args:
            ev: The StartEvent of the workflow.
            query: The user query string.
            top_k: The number of top documents/contexts to retrieve.
            rerank_top_n: The number of documents to keep after re-ranking (if enabled).
        Returns:
            A RetrievalResult event containing the retrieved nodes and metadata.
        """
        try:
            query_bundle = self._prepare_query(query)
            st.info(f"📚 Retrieving information for: '{query_bundle.query_str}' using {type(self.index).__name__}")
            logger.info(f"Retrieving for query: '{query_bundle.query_str}' with top_k={top_k}, index_type={type(self.index).__name__}")
            
            if not self.index:
                logger.error("Index is not loaded for retrieval.")
                raise ValueError("Index is not loaded")

            nodes: List[NodeWithScore] = []
            retrieval_metadata: Dict[str, Any] = {} # Specific metadata for this retrieval

            # --- Vector Store Index Retrieval ---
            if isinstance(self.index, VectorStoreIndex):
                logger.info("Using VectorStoreIndex for retrieval.")
                retriever = self.index.as_retriever(similarity_top_k=top_k)
                nodes = await retriever.aretrieve(query_bundle) # Pass QueryBundle
                
                # Apply similarity cutoff postprocessor for vector search
                similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
                nodes = similarity_postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
                
                retrieval_metadata["retrieval_type"] = "vector_search"
                retrieval_metadata["similarity_cutoff_applied"] = self.similarity_cutoff
                logger.info(f"Vector retrieval found {len(nodes)} nodes after similarity cutoff.")

            # --- Knowledge Graph Index Retrieval ---
            elif isinstance(self.index, KnowledgeGraphIndex):
                logger.info("Using KnowledgeGraphIndex for retrieval.")
                graph_retriever_mode = st.session_state.get("graph_retriever_mode", "keyword")
                
                # KG retriever can use different modes.
                # It typically retrieves text chunks representing graph knowledge or subgraphs.
                kg_retriever = self.index.as_retriever(
                    retriever_mode=graph_retriever_mode, 
                    similarity_top_k=top_k, # Used for 'embedding' or 'hybrid' modes
                    include_text=True, # Ensure text is part of the node content
                    # graph_store_query_depth=st.session_state.get("graph_traversal_depth", 2) # Example if using depth
                )
                # KnowledgeGraphQueryEngine might be more flexible for complex graph queries
                # query_engine = RetrieverQueryEngine.from_args(retriever=kg_retriever, llm=self.llm)
                # response = await query_engine.aquery(query_bundle.query_str)
                # nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
                nodes = await kg_retriever.aretrieve(query_bundle.query_str) # Pass string query to KG retriever
                
                retrieval_metadata["retrieval_type"] = "graph_search"
                retrieval_metadata["graph_retriever_mode"] = graph_retriever_mode
                logger.info(f"Graph retrieval (mode: {graph_retriever_mode}) found {len(nodes)} nodes.")
            else:
                logger.error(f"Unsupported index type: {type(self.index)}")
                raise ValueError(f"Unsupported index type: {type(self.index)}")

            # --- Common Metadata and Re-ranking ---
            base_metadata = {
                "total_nodes_pre_rerank": len(nodes),
                "query_embedding_model": Settings.embed_model.__class__.__name__ if Settings.embed_model else "N/A",
                **retrieval_metadata # Merge specific retrieval metadata
            }

            # Apply re-ranking if enabled and nodes are suitable (have text content)
            if st.session_state.get("enable_reranker") and nodes:
                # Ensure nodes have text for the reranker
                text_nodes = [n for n in nodes if hasattr(n, 'text') and isinstance(n.text, str) and n.text.strip()]
                if text_nodes:
                    try:
                        logger.info(f"Applying reranker with top_n={rerank_top_n}.")
                        nodes = functional_reranker(text_nodes, query_bundle.query_str, rerank_top_n)
                        base_metadata["reranked"] = True
                        base_metadata["rerank_top_n_used"] = rerank_top_n
                        logger.info(f"Reranking resulted in {len(nodes)} nodes.")
                    except Exception as e:
                        logger.warning(f"Reranking failed: {str(e)}", exc_info=True)
                        base_metadata["reranking_error"] = str(e)
                else:
                    logger.info("Skipping reranker as no suitable text nodes found.")
                    base_metadata["reranker_skipped"] = "No text nodes for reranking"
            
            base_metadata["total_nodes_post_processing"] = len(nodes)

            # Process node texts for consistent encoding and stripping
            for node_with_score in nodes: 
                if hasattr(node_with_score, 'node') and hasattr(node_with_score.node, 'text'):
                    node_with_score.node.text = self._process_text(node_with_score.node.text)
                elif hasattr(node_with_score, 'text'): # If node is directly NodeWithScore without nested .node
                     node_with_score.text = self._process_text(node_with_score.text)

            return RetrievalResult(
                nodes=nodes,
                original_query=query_bundle.query_str,
                metadata=base_metadata
            )

        except Exception as e:
            logger.error(f"Retrieval step error: {str(e)}", exc_info=True)
            # Return an empty result with error information in metadata
            return RetrievalResult(nodes=[], original_query=query, metadata={"error": str(e), "details": "Error in retrieval step"})

    @step
    async def synthesize(self, ev: RetrievalResult) -> StopEvent:
        """
        Response synthesis step: Generates a response based on the retrieved context.
        Args:
            ev: The RetrievalResult event containing nodes and query.
        Returns:
            A StopEvent containing the final response, judgement, and metadata.
        """
        if not ev.nodes:
            logger.info("No relevant information found to synthesize a response.")
            return StopEvent(result={
                "response": "No relevant information was found based on your query and current retrieval settings.",
                "judgement": "N/A (No response to judge)",
                "metadata": ev.metadata # Pass along retrieval metadata
            })

        try:
            logger.info(f"Synthesizing response for query: '{ev.original_query}' with {len(ev.nodes)} nodes.")
            
            # Configure the response synthesizer.
            # `get_response_synthesizer` is flexible. "compact" is a good default.
            response_synthesizer = get_response_synthesizer(
                llm=self.llm, # LLM is already set via Settings.llm by __init__
                response_mode="compact", # Other modes: "refine", "tree_summarize", "simple_summarize"
                # Example of custom prompts (optional):
                # text_qa_template=PromptTemplate("Context: {context_str}\nQuery: {query_str}\nAnswer:"),
                # refine_template=PromptTemplate("..."),
                verbose=True, # Logs details of the synthesis process
                use_async=True # Ensures async operation
            )
            
            # Generate response
            # The synthesizer expects the query string and list of NodeWithScore objects.
            response_obj = await response_synthesizer.asynthesize(
                query=ev.original_query,
                nodes=ev.nodes
            )
            
            if not response_obj or not hasattr(response_obj, 'response'):
                logger.error("Invalid response format from LLM or synthesizer.")
                raise ValueError("Invalid response format from LLM or synthesizer")
            
            response_text = self._process_text(str(response_obj.response))
            if not response_text:
                logger.warning("Response processing resulted in empty text.")
                # Consider this a partial failure or handle as needed
                response_text = "The language model generated an empty response."


            # --- LLM-as-a-Judge Evaluation (if enabled) ---
            judgement_text = "LLM-as-a-Judge disabled."
            if st.session_state.get("enable_llm_judge") and st.session_state.get("groq_api_key_sidebar"):
                logger.info("Performing LLM-as-a-Judge evaluation.")
                try:
                    raw_judgement = await functional_llm_as_judge(
                        query=ev.original_query,
                        response_text=response_text,
                        judge_llm_name=st.session_state.get("selected_model"), # Use the same model or a dedicated judge model
                        groq_api_key=st.session_state["groq_api_key_sidebar"]
                    )
                    judgement_text = self._process_text(str(raw_judgement)) if raw_judgement else "Judge evaluation produced no result or an empty result."
                    logger.info(f"Judge evaluation result: {judgement_text[:100]}...") # Log snippet
                except Exception as e:
                    logger.error(f"Judge evaluation error: {str(e)}", exc_info=True)
                    judgement_text = f"Judge evaluation failed: {str(e)}"
            else:
                logger.info("LLM-as-a-Judge is disabled or API key is missing.")


            # --- Final Metadata Update ---
            synthesis_metadata = {
                "response_length_chars": len(response_text),
                "llm_used_for_synthesis": self.llm.__class__.__name__,
                "response_synthesizer_mode": "compact", # Or reflect actual mode used
                "has_judgement": judgement_text != "LLM-as-a-Judge disabled." and not judgement_text.startswith("Judge evaluation failed")
            }
            final_metadata = {**ev.metadata, **synthesis_metadata} # Combine retrieval and synthesis metadata

            return StopEvent(result={
                "response": response_text,
                "judgement": judgement_text,
                "metadata": final_metadata
            })

        except Exception as e:
            logger.error(f"Synthesis step error: {str(e)}", exc_info=True)
            return StopEvent(result={
                "response": f"Error during response generation: {str(e)}",
                "judgement": "N/A (Error in synthesis)",
                "metadata": {**ev.metadata, "synthesis_error": str(e), "details": "Error in synthesis step"}
            })
