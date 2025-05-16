import streamlit as st
import asyncio
from config import NOMIC_EMBED_MODEL
from utils.llm_utils import get_llm, get_embedding_model
from workflow.rag_workflow import RAGWorkflow, StartEvent


def handle_main_ui():
    query = st.text_area("Enter your query:", height=100, key="query_input")

    if st.button("Run RAG Workflow"):
        if not query:
            st.warning("Please enter a query.")
            return

        if "vector_index" not in st.session_state:
            st.error("Index not loaded. Build/load it from sidebar.")
            return

        llm = get_llm(st.session_state.selected_model, st.session_state["groq_api_key_sidebar"])
        embed_model = get_embedding_model()

        rag = RAGWorkflow(llm=llm, embed_model=embed_model, vector_index=st.session_state.vector_index)

        with st.spinner("Running RAG Workflow..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ev = StartEvent(query=query, top_k=st.session_state.top_k, rerank_top_n=st.session_state.get("rerank_top_n", 3))
                retrieved = loop.run_until_complete(rag.retrieve(ev, query=query, top_k=st.session_state.top_k, rerank_top_n=st.session_state.get("rerank_top_n", 3)))
                if retrieved and retrieved.nodes:
                    result = loop.run_until_complete(rag.synthesize(retrieved))
                    st.subheader(f"\ud83d\udcac Response from {st.session_state.selected_model}:")
                    st.markdown(result.result["response"])
                    if st.session_state.enable_llm_judge:
                        st.subheader("LLM-as-a-Judge Evaluation:")
                        st.markdown(result.result["judgement"])
                else:
                    st.warning("No relevant documents retrieved.")
            except Exception as e:
                st.error(f"Workflow execution failed: {e}")
