# LLM/workflow/rag_workflow.py
# Defines the RAG (Retrieval Augmented Generation) workflow using LlamaIndex.

import streamlit as st
from typing import List, Optional, Dict, Any, Tuple
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import (
    KnowledgeGraphIndex,
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    QueryBundle
)
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.prompts import PromptTemplate as LlamaPromptTemplate
from llama_index.core.llms import LLM

from utils.rerank_utils import functional_reranker
from utils.judge_utils import functional_llm_as_judge
from config import DEFAULT_QUERY_TRANSFORMATION_PROMPT # Import the new prompt

import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryTransformEvent(Event):
    original_query: str
    transformed_query: str
    metadata: Dict[str, Any]

class RetrievalResult(Event):
    nodes: List[NodeWithScore]
    original_query: str
    transformed_query: Optional[str] # Added
    metadata: Dict[str, Any]

class RAGWorkflow(Workflow):
    def __init__(self, llm: LLM, index: Any, documents: Optional[List[TextNode]] = None, **kwargs): # Added documents
        super().__init__(**kwargs)
        self.llm = llm
        self.index = index
        self.documents = documents # Store documents for potential BM25
        self.similarity_cutoff = 0.7
        self.max_tokens_for_llm = 4096

        if Settings.llm is None or Settings.llm != self.llm:
            Settings.llm = self.llm
            logger.info(f"RAGWorkflow: Settings.llm updated to {self.llm.__class__.__name__}")

    def _process_text(self, text: Any) -> str:
        if not isinstance(text, str):
            text = str(text)
        try:
            text = text.encode('utf-8', 'ignore').decode('utf-8')
            return text.strip()
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}", exc_info=True)
            return ""

    def _prepare_query(self, query: str) -> QueryBundle:
        processed_query = self._process_text(query)
        if not processed_query:
            logger.error("Query is empty after processing.")
            raise ValueError("Query is empty after processing")
        return QueryBundle(query_str=processed_query)

    @step
    async def transform_query(self, ev: StartEvent, query: str) -> QueryTransformEvent:
        """
        Transforms the query using an LLM if enabled.
        """
        original_query = query
        transformed_query = query # Default to original if not transformed
        metadata = {"transformation_applied": False}

        if st.session_state.get("enable_query_transformation", False):
            mode = st.session_state.get("query_transformation_mode", "Default Expansion")
            logger.info(f"Transforming query (mode: {mode}): '{query}'")
            
            try:
                if mode == "Hypothetical Document (HyDE)":
                    hyde_prompt_template = LlamaPromptTemplate(
                        "Please write a short, hypothetical passage that could answer the following user query. "
                        "This passage will be used to retrieve relevant documents.\n"
                        "User Query: {query_str}\nPassage:"
                    )
                    hyde_pipeline = QueryPipeline(
                        chain=[hyde_prompt_template, self.llm], # self.llm is the LLM instance
                    )
                    
                    # The pipeline's arun method will execute the chain.
                    # The output 'hyde_response_obj' will be the result from the last component (self.llm).
                    hyde_response_obj = await hyde_pipeline.arun(query_str=query)
                    
                    actual_generated_text: Optional[str] = None

                    # Extract text based on common LlamaIndex response types
                    if isinstance(hyde_response_obj, str):
                        # If the pipeline already processed it to a string
                        actual_generated_text = hyde_response_obj
                    elif hasattr(hyde_response_obj, 'message') and hasattr(hyde_response_obj.message, 'content'):
                        # This is the standard way to get content from a LlamaIndex ChatResponse
                        actual_generated_text = hyde_response_obj.message.content
                    elif hasattr(hyde_response_obj, 'text'):
                        # Fallback for CompletionResponse or other similar response objects
                        actual_generated_text = hyde_response_obj.text
                    
                    if actual_generated_text is None:
                        logger.error(f"Query transformation (HyDE) failed to extract text from response of type: {type(hyde_response_obj)}. Full response: {hyde_response_obj}")
                        transformed_query = query # Fallback to original query
                        metadata["transformation_error"] = f"HyDE response type unhandled: {type(hyde_response_obj)}"
                        metadata["transformation_applied"] = False # Ensure this reflects failure
                    else:
                        transformed_query = self._process_text(actual_generated_text)
                        st.info(f"🧠 HyDE Query (using hypothetical doc): {transformed_query[:200]}...") # Show a bit more for context
                        metadata["transformation_details"] = f"HyDE generated document of length {len(transformed_query)}"
                        metadata["transformation_applied"] = True # Mark as successful if text was extracted
                        metadata["transformation_mode"] = mode


                else: # Default Expansion (or other modes if you add them)
                    prompt = DEFAULT_QUERY_TRANSFORMATION_PROMPT.format(original_query=query)
                    response = await self.llm.acomplete(prompt) # Assuming .acomplete for completion LLMs
                    
                    actual_generated_text: Optional[str] = None
                    if hasattr(response, 'text'):
                        actual_generated_text = response.text
                    elif hasattr(response, 'message') and hasattr(response.message, 'content'): # Just in case acomplete returns chat-like
                        actual_generated_text = response.message.content

                    if actual_generated_text is None:
                        logger.error(f"Query transformation (Default Expansion) failed to extract text from response: {type(response)}. Full response: {response}")
                        transformed_query = query # Fallback to original query
                        metadata["transformation_error"] = f"Default Expansion response type unhandled: {type(response)}"
                        metadata["transformation_applied"] = False
                    else:
                        transformed_query = self._process_text(actual_generated_text)
                        st.info(f"🧠 Transformed Query: {transformed_query}")
                        metadata["transformation_details"] = f"LLM expansion. Original length: {len(query)}, New length: {len(transformed_query)}"
                        metadata["transformation_applied"] = True
                        metadata["transformation_mode"] = mode
            
            except Exception as e:
                logger.error(f"Query transformation failed with exception: {str(e)}", exc_info=True)
                transformed_query = query # Fallback to original query on any exception
                metadata["transformation_error"] = str(e)
                metadata["transformation_applied"] = False # Mark as failed
        
        return QueryTransformEvent(
            original_query=original_query,
            transformed_query=transformed_query,
            metadata=metadata
        )

    @step
    async def retrieve(self, ev: QueryTransformEvent, top_k: int = 3, rerank_top_n: int = 3) -> RetrievalResult:
        original_query = ev.original_query
        query_to_use = ev.transformed_query # Use transformed query
        query_bundle = self._prepare_query(query_to_use)
        
        search_mode = st.session_state.get("search_mode", "Vector Only")
        sparse_top_k = st.session_state.get("sparse_top_k", 3) if search_mode == "Hybrid (Vector + Sparse Fusion)" else 0

        st.info(f"📚 Retrieving (mode: {search_mode}) for: '{query_to_use}' (Original: '{original_query}')")
        logger.info(f"Retrieving for query: '{query_bundle.query_str}' (orig: '{original_query}') "
                    f"with dense_top_k={top_k}, sparse_top_k={sparse_top_k}, index_type={type(self.index).__name__}, search_mode='{search_mode}'")

        if not self.index:
            logger.error("Index is not loaded for retrieval.")
            raise ValueError("Index is not loaded")

        nodes: List[NodeWithScore] = []
        retrieval_metadata: Dict[str, Any] = ev.metadata # Carry over transformation metadata

        # --- Vector Store Index Retrieval ---
        if isinstance(self.index, VectorStoreIndex):
            logger.info("Using VectorStoreIndex for dense retrieval.")
            vector_retriever = self.index.as_retriever(similarity_top_k=top_k)
            dense_nodes = await vector_retriever.aretrieve(query_bundle)
            nodes.extend(dense_nodes)
            retrieval_metadata["dense_retrieval_count"] = len(dense_nodes)

            if search_mode == "Hybrid (Vector + Sparse Fusion)":
                st.info(f"Performing sparse retrieval (simulated/BM25) with top_k={sparse_top_k}...")
                if self.documents: # Check if documents were passed for BM25
                    try:
                        # Ensure NLTK resources for tokenization if not already downloaded
                        # import nltk
                        # try:
                        #     nltk.data.find('tokenizers/punkt')
                        # except nltk.downloader.DownloadError:
                        #     nltk.download('punkt', quiet=True)
                        
                        bm25_retriever = BM25Retriever.from_defaults(
                            nodes=self.documents, # Use pre-loaded documents/nodes
                            similarity_top_k=sparse_top_k
                        )
                        sparse_nodes = await bm25_retriever.aretrieve(query_bundle.query_str) # BM25Retriever takes string
                        # Deduplicate nodes based on node_id if combining
                        existing_node_ids = {n.node.node_id for n in nodes}
                        for sn in sparse_nodes:
                            if sn.node.node_id not in existing_node_ids:
                                nodes.append(sn)
                                existing_node_ids.add(sn.node.node_id)
                        retrieval_metadata["sparse_retrieval_count"] = len(sparse_nodes)
                        logger.info(f"BM25 retrieval found {len(sparse_nodes)} nodes. Combined unique nodes: {len(nodes)}")
                    except Exception as e:
                        logger.error(f"BM25 retrieval failed: {str(e)}", exc_info=True)
                        st.warning(f"BM25 part of hybrid search failed: {e}")
                        retrieval_metadata["sparse_retrieval_error"] = str(e)
                else:
                    logger.warning("Hybrid search selected but no documents available for BM25Retriever.")
                    st.warning("Hybrid search: Documents not loaded for sparse retrieval part.")
            
            # Apply similarity cutoff postprocessor for vector search (applies to all nodes for simplicity here)
            similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
            nodes = similarity_postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
            retrieval_metadata["similarity_cutoff_applied"] = self.similarity_cutoff

        # --- Knowledge Graph Index Retrieval ---
        elif isinstance(self.index, KnowledgeGraphIndex):
            logger.info("Using KnowledgeGraphIndex for retrieval.")
            graph_retriever_mode = st.session_state.get("graph_retriever_mode", "keyword")
            graph_traversal_depth = st.session_state.get("graph_traversal_depth", 2)
            
            kg_retriever = self.index.as_retriever(
                retriever_mode=graph_retriever_mode,
                similarity_top_k=top_k, # Used for 'embedding' or 'hybrid' KG modes
                include_text=True,
                graph_store_query_depth=graph_traversal_depth # Added depth
            )
            nodes = await kg_retriever.aretrieve(query_bundle.query_str)
            retrieval_metadata["graph_retriever_mode"] = graph_retriever_mode
            retrieval_metadata["graph_traversal_depth"] = graph_traversal_depth
        else:
            logger.error(f"Unsupported index type: {type(self.index)}")
            raise ValueError(f"Unsupported index type: {type(self.index)}")

        retrieval_metadata["total_nodes_pre_rerank"] = len(nodes)
        retrieval_metadata["query_embedding_model"] = Settings.embed_model.__class__.__name__ if Settings.embed_model else "N/A"

        if st.session_state.get("enable_reranker", True) and nodes:
            text_nodes = [n for n in nodes if hasattr(n, 'text') and isinstance(n.text, str) and n.text.strip()]
            if text_nodes:
                try:
                    logger.info(f"Applying reranker with top_n={rerank_top_n}.")
                    nodes = functional_reranker(text_nodes, original_query, rerank_top_n) # Rerank on original query
                    retrieval_metadata["reranked"] = True
                    retrieval_metadata["rerank_top_n_used"] = rerank_top_n
                except Exception as e:
                    logger.warning(f"Reranking failed: {str(e)}", exc_info=True)
                    retrieval_metadata["reranking_error"] = str(e)
            else:
                base_metadata["reranker_skipped"] = "No text nodes for reranking"
        
        retrieval_metadata["total_nodes_post_processing"] = len(nodes)

        for node_with_score in nodes:
            if hasattr(node_with_score, 'node') and hasattr(node_with_score.node, 'text'):
                node_with_score.node.text = self._process_text(node_with_score.node.text)
            elif hasattr(node_with_score, 'text'):
                 node_with_score.text = self._process_text(node_with_score.text)

        return RetrievalResult(
            nodes=nodes,
            original_query=original_query,
            transformed_query=query_to_use if ev.metadata.get("transformation_applied") else None,
            metadata=retrieval_metadata
        )

    @step
    async def synthesize(self, ev: RetrievalResult) -> StopEvent:
        if not ev.nodes:
            logger.info("No relevant information found to synthesize a response.")
            response_text = "No relevant information was found based on your query and current retrieval settings."
            if ev.transformed_query:
                response_text += f"\n(Query after transformation: '{ev.transformed_query}')"
            return StopEvent(result={
                "response": response_text,
                "judgement": "N/A (No response to judge)",
                "metadata": ev.metadata
            })

        try:
            query_for_synthesis = ev.original_query # Always use original query for final answer synthesis
            logger.info(f"Synthesizing response for query: '{query_for_synthesis}' with {len(ev.nodes)} nodes.")
            
            response_synthesizer = get_response_synthesizer(
                llm=self.llm,
                response_mode="compact",
                verbose=True,
                use_async=True
            )
            
            response_obj = await response_synthesizer.asynthesize(
                query=query_for_synthesis,
                nodes=ev.nodes
            )
            
            if not response_obj or not hasattr(response_obj, 'response'):
                logger.error("Invalid response format from LLM or synthesizer.")
                raise ValueError("Invalid response format from LLM or synthesizer")
            
            response_text = self._process_text(str(response_obj.response))
            if not response_text:
                response_text = "The language model generated an empty response."

            judgement_text = "LLM-as-a-Judge disabled."
            if st.session_state.get("enable_llm_judge") and st.session_state.get("groq_api_key_sidebar"):
                logger.info("Performing LLM-as-a-Judge evaluation.")
                try:
                    raw_judgement = await functional_llm_as_judge(
                        query=query_for_synthesis, # Judge based on original query
                        response_text=response_text,
                        judge_llm_name=st.session_state.get("selected_model"),
                        groq_api_key=st.session_state["groq_api_key_sidebar"]
                    )
                    judgement_text = self._process_text(str(raw_judgement)) if raw_judgement else "Judge evaluation produced no result."
                except Exception as e:
                    logger.error(f"Judge evaluation error: {str(e)}", exc_info=True)
                    judgement_text = f"Judge evaluation failed: {str(e)}"
            
            synthesis_metadata = {
                "response_length_chars": len(response_text),
                "llm_used_for_synthesis": self.llm.__class__.__name__,
                "response_synthesizer_mode": "compact",
                "has_judgement": judgement_text != "LLM-as-a-Judge disabled." and not judgement_text.startswith("Judge evaluation failed")
            }
            if ev.transformed_query:
                synthesis_metadata["transformed_query_used_for_retrieval"] = ev.transformed_query

            final_metadata = {**ev.metadata, **synthesis_metadata}

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

    def _get_bm25_retriever(self, sparse_top_k: int) -> Optional[BM25Retriever]:
        """Helper to initialize BM25Retriever if documents are available."""
        if self.documents:
            try:
                # This is a simplified initialization. You might need to customize tokenization.
                # Ensure NLTK punkt is available if BM25Retriever's default tokenizer needs it.
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except nltk.downloader.DownloadError:
                    logger.info("NLTK 'punkt' not found, downloading...")
                    nltk.download('punkt', quiet=True)
                
                return BM25Retriever.from_defaults(
                    nodes=self.documents, 
                    similarity_top_k=sparse_top_k
                )
            except Exception as e:
                logger.error(f"Failed to initialize BM25Retriever: {e}")
                st.warning(f"Could not initialize BM25Retriever: {e}")
        return None