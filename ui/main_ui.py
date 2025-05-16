# LLM/ui/main_ui.py
# This file handles the main user interface elements and logic for the RAG application.

import streamlit as st
import asyncio
from utils.llm_utils import get_llm
from workflow.rag_workflow import RAGWorkflow, StartEvent, RetrievalResult, QueryTransformEvent # Added QueryTransformEvent
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_metadata(metadata: dict):
    if metadata:
        with st.expander("View Processing Details (Metadata)", expanded=False):
            try:
                st.json(json.loads(json.dumps(metadata, default=str)), expanded=True)
            except Exception as e:
                logger.warning(f"Could not pretty-print metadata, displaying as is: {e}")
                st.write(metadata)
    else:
        st.info("No metadata available to display.")

def handle_main_ui():
    st.header("💬 Query Your Documents")
    query = st.text_area(
        "Enter your query:",
        height=100,
        key="query_input",
        placeholder="Ask something about the content of your indexed documents..."
    )

    if st.button("🚀 Run RAG Workflow", key="run_rag_button"):
        if not query:
            st.warning("⚠️ Please enter a query.")
            return

        if "groq_api_key_sidebar" not in st.session_state or not st.session_state["groq_api_key_sidebar"]:
            st.error("❌ Groq API Key is missing. Please provide it in the sidebar.")
            logger.warning("Groq API key missing, cannot run RAG workflow.")
            return

        current_index = None
        selected_index_type = st.session_state.get("selected_index_type")

        if selected_index_type == "Knowledge Graph Index" and "knowledge_graph_index" in st.session_state:
            current_index = st.session_state.knowledge_graph_index
            logger.info("Using Knowledge Graph Index for RAG workflow.")
        elif selected_index_type == "Vector Index" and "vector_index" in st.session_state:
            current_index = st.session_state.vector_index
            logger.info("Using Vector Index for RAG workflow.")

        if not current_index:
            st.error(
                f"❌ The selected index type ('{selected_index_type or 'N/A'}') "
                "is not built or loaded. Please build it from the sidebar."
            )
            logger.warning(f"Index not loaded for type: {selected_index_type}")
            return

        try:
            llm = get_llm(
                st.session_state.selected_model,
                st.session_state.get("groq_api_key_sidebar", "")
            )
            if not llm:
                st.error("❌ LLM could not be initialized. Please check API key and model selection in the sidebar.")
                logger.error("LLM initialization failed.")
                return

            # Pass loaded documents to RAGWorkflow if available (for BM25 in hybrid)
            loaded_documents = st.session_state.get("loaded_documents", None)
            rag_workflow_instance = RAGWorkflow(llm=llm, index=current_index, documents=loaded_documents)
            logger.info(f"RAGWorkflow initialized with LLM: {llm.__class__.__name__}, "
                        f"Index: {current_index.__class__.__name__}, "
                        f"Documents loaded: {bool(loaded_documents)}")


            async def run_async_rag_steps():
                top_k_retrieval = st.session_state.get("top_k", 3)
                rerank_top_n_after_retrieval = st.session_state.get("rerank_top_n", 3)
                
                # Start with the original query
                start_event = StartEvent(input=query) # Pass query as input to the event for the first step

                logger.info(f"Calling RAGWorkflow.transform_query for query: '{query}'")
                transform_event_data: QueryTransformEvent = await rag_workflow_instance.transform_query(
                    ev=start_event, query=query # Pass query explicitly to the first custom step
                )

                if transform_event_data.metadata.get("transformation_error"):
                    st.warning(f"Query transformation error: {transform_event_data.metadata['transformation_error']}")

                logger.info(f"Calling RAGWorkflow.retrieve with transformed_query: '{transform_event_data.transformed_query}'")
                retrieval_event_data: RetrievalResult = await rag_workflow_instance.retrieve(
                    ev=transform_event_data, # Pass the event from the previous step
                    top_k=top_k_retrieval, 
                    rerank_top_n=rerank_top_n_after_retrieval
                )

                synthesis_event_data = None

                if retrieval_event_data.metadata and "error" in retrieval_event_data.metadata:
                    st.error(f"Retrieval error: {retrieval_event_data.metadata['error']}")
                    logger.error(f"Retrieval error indicated in metadata: {retrieval_event_data.metadata}")
                    return retrieval_event_data, None

                if retrieval_event_data.nodes:
                    logger.info(f"Calling RAGWorkflow.synthesize with {len(retrieval_event_data.nodes)} nodes.")
                    synthesis_event_data = await rag_workflow_instance.synthesize(retrieval_event_data)
                else:
                    st.warning("⚠️ No relevant documents or graph contexts found for your query. Try rephrasing or adjusting RAG parameters.")
                    logger.warning("No nodes found by retrieval, skipping synthesis.")
                
                return retrieval_event_data, synthesis_event_data

            with st.spinner("🔄 Running RAG Workflow... This may take a moment."):
                try:
                    retrieval_result, synthesis_result = asyncio.run(run_async_rag_steps())

                    if retrieval_result:
                        if not (retrieval_result.metadata and "error" in retrieval_result.metadata):
                             num_items_retrieved = retrieval_result.metadata.get('total_nodes_post_processing', len(retrieval_result.nodes))
                             retrieval_query_info = f"'{retrieval_result.original_query}'"
                             if retrieval_result.transformed_query and retrieval_result.metadata.get("transformation_applied"):
                                 retrieval_query_info += f" (Transformed: '{retrieval_result.transformed_query}')"
                             
                             st.info(
                                f"Retrieved {num_items_retrieved} items for query: {retrieval_query_info}"
                            )

                    if synthesis_result and synthesis_result.result:
                        final_response_data = synthesis_result.result
                        
                        st.subheader(f"💬 Response from {st.session_state.selected_model}:")
                        st.markdown(final_response_data.get("response", "No response content."), unsafe_allow_html=True)
                        logger.info(f"Response generated. Length: {len(final_response_data.get('response', ''))} chars.")

                        if st.session_state.get("enable_llm_judge") and final_response_data.get("judgement"):
                            st.subheader("⚖️ LLM-as-a-Judge Evaluation:")
                            st.markdown(final_response_data.get("judgement", "No judgement content."), unsafe_allow_html=True)
                            logger.info("LLM Judge evaluation displayed.")
                        
                        if final_response_data.get("metadata"):
                            display_metadata(final_response_data["metadata"])
                        
                    elif retrieval_result and retrieval_result.metadata:
                         display_metadata(retrieval_result.metadata)

                except Exception as e:
                    logger.error(f"Error during RAG workflow execution or result handling: {str(e)}", exc_info=True)
                    st.error(f"❌ An error occurred during RAG workflow execution: {str(e)}")

        except Exception as e:
            logger.error(f"Error setting up RAG workflow: {str(e)}", exc_info=True)
            st.error(f"❌ Error setting up the RAG workflow: {str(e)}")