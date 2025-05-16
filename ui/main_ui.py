# LLM/ui/main_ui.py
# This file handles the main user interface elements and logic for the RAG application.

import streamlit as st
import asyncio # Ensure asyncio is imported
# from config import NOMIC_EMBED_MODEL # Not directly used in this file's logic
from utils.llm_utils import get_llm
from workflow.rag_workflow import RAGWorkflow, StartEvent, RetrievalResult # Import RetrievalResult for type hinting
import logging
import json # For pretty printing metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_metadata(metadata: dict):
    """
    Displays workflow metadata in a structured and expandable JSON format.
    Args:
        metadata: A dictionary containing metadata from the RAG workflow.
    """
    if metadata:
        with st.expander("View Processing Details (Metadata)", expanded=False):
            try:
                # Attempt to pretty-print the JSON for better readability
                st.json(json.loads(json.dumps(metadata, default=str)), expanded=True)
            except Exception as e:
                logger.warning(f"Could not pretty-print metadata, displaying as is: {e}")
                st.write(metadata) # Fallback to direct write if JSON conversion fails
    else:
        st.info("No metadata available to display.")


def handle_main_ui():
    """
    Handles the main UI elements: query input, run button, and display of results.
    Uses asyncio.run() to correctly call asynchronous workflow steps.
    """
    st.header("💬 Query Your Documents")
    query = st.text_area(
        "Enter your query:", 
        height=100, 
        key="query_input",
        placeholder="Ask something about the content of your indexed documents..."
    )

    if st.button("🚀 Run RAG Workflow", key="run_rag_button"):
        # --- Initial Synchronous Checks ---
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
            # --- Initialize LLM and RAG Workflow (Synchronous Part) ---
            llm = get_llm(
                st.session_state.selected_model, 
                st.session_state.get("groq_api_key_sidebar", "")
            )
            if not llm:
                st.error("❌ LLM could not be initialized. Please check API key and model selection in the sidebar.")
                logger.error("LLM initialization failed.")
                return

            rag_workflow_instance = RAGWorkflow(llm=llm, index=current_index)
            logger.info(f"RAGWorkflow initialized with LLM: {llm.__class__.__name__} and Index: {current_index.__class__.__name__}")

            # --- Define inner async function to run the asynchronous workflow steps ---
            async def run_async_rag_steps():
                top_k_retrieval = st.session_state.get("top_k", 3)
                rerank_top_n_after_retrieval = st.session_state.get("rerank_top_n", 3)

                start_event = StartEvent(
                    query=query, top_k=top_k_retrieval, rerank_top_n=rerank_top_n_after_retrieval
                )

                logger.info(f"Calling RAGWorkflow.retrieve for query: '{query}'")
                # Type hint for clarity
                retrieval_event_data: RetrievalResult = await rag_workflow_instance.retrieve(
                    ev=start_event, query=query, top_k=top_k_retrieval, rerank_top_n=rerank_top_n_after_retrieval
                )
                
                synthesis_event_data = None # Initialize to None

                # Check for errors reported by the retrieval step itself in its metadata
                if retrieval_event_data.metadata and "error" in retrieval_event_data.metadata:
                    st.error(f"Retrieval error: {retrieval_event_data.metadata['error']}")
                    logger.error(f"Retrieval error indicated in metadata: {retrieval_event_data.metadata}")
                    # No synthesis if retrieval had a critical error, return what we have
                    return retrieval_event_data, None

                # Proceed to synthesis if nodes were found and no retrieval error
                if retrieval_event_data.nodes:
                    logger.info(f"Calling RAGWorkflow.synthesize with {len(retrieval_event_data.nodes)} nodes.")
                    synthesis_event_data = await rag_workflow_instance.synthesize(retrieval_event_data)
                else:
                    # This case means retrieval was successful (no error in metadata) but found no nodes.
                    st.warning("⚠️ No relevant documents or graph contexts found for your query. Try rephrasing or adjusting RAG parameters.")
                    logger.warning("No nodes found by retrieval, skipping synthesis.")
                
                return retrieval_event_data, synthesis_event_data

            # --- Execute the async parts within the button click handler using asyncio.run() ---
            with st.spinner("🔄 Running RAG Workflow... This may take a moment."):
                try:
                    # asyncio.run() will create and manage its own event loop for the async function.
                    retrieval_result, synthesis_result = asyncio.run(run_async_rag_steps())

                    # --- Display Results ---

                    # Display information based on retrieval_result, unless it indicated a critical error
                    if retrieval_result:
                        # Show "Retrieved X items" info only if retrieval didn't self-report a critical error
                        if not (retrieval_result.metadata and "error" in retrieval_result.metadata):
                             num_items_retrieved = retrieval_result.metadata.get('total_nodes_post_processing', len(retrieval_result.nodes))
                             st.info(
                                f"Retrieved {num_items_retrieved} "
                                f"items for query: '{retrieval_result.original_query}'"
                            )
                        # The full metadata will be displayed later, either combined with synthesis or by itself.

                    # Handle and display synthesis result if available
                    if synthesis_result and synthesis_result.result:
                        final_response_data = synthesis_result.result
                        
                        st.subheader(f"💬 Response from {st.session_state.selected_model}:")
                        st.markdown(final_response_data.get("response", "No response content."), unsafe_allow_html=True)
                        logger.info(f"Response generated. Length: {len(final_response_data.get('response', ''))} chars.")

                        if st.session_state.get("enable_llm_judge") and final_response_data.get("judgement"):
                            st.subheader("⚖️ LLM-as-a-Judge Evaluation:")
                            st.markdown(final_response_data.get("judgement", "No judgement content."), unsafe_allow_html=True)
                            logger.info("LLM Judge evaluation displayed.")
                        
                        # Display combined metadata from synthesis (which includes retrieval metadata)
                        if final_response_data.get("metadata"):
                            display_metadata(final_response_data["metadata"])
                        
                    elif retrieval_result: # Synthesis didn't happen or produced no result, but retrieval did run
                        # If retrieval had an error, its message was already shown.
                        # If retrieval was OK but no nodes (warning already shown), display retrieval metadata.
                        # If synthesis failed after successful retrieval, display retrieval metadata.
                        if retrieval_result.metadata:
                             display_metadata(retrieval_result.metadata)
                    # If retrieval_result itself is None (shouldn't happen with current rag_workflow.py design,
                    # as it always returns a RetrievalResult), then an error occurred very early in run_async_rag_steps.

                except Exception as e: # Catch errors from asyncio.run() or synchronous result handling
                    logger.error(f"Error during RAG workflow execution or result handling: {str(e)}", exc_info=True)
                    st.error(f"❌ An error occurred during RAG workflow execution: {str(e)}")

        except Exception as e: # Catch errors from initial synchronous setup (LLM/workflow instantiation)
            logger.error(f"Error setting up RAG workflow: {str(e)}", exc_info=True)
            st.error(f"❌ Error setting up the RAG workflow: {str(e)}")
