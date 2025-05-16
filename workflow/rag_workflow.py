
import streamlit as st
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, Event, step
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from utils.rerank_utils import functional_reranker
from utils.judge_utils import functional_llm_as_judge

class RetrieverEvent(Event):
    nodes: list[NodeWithScore]
    original_query: str

class RAGWorkflow(Workflow):
    def __init__(self, llm, embed_model, vector_index, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.embed_model = embed_model
        self.vector_index = vector_index

    @step
    async def retrieve(self, ev: StartEvent, query: str, top_k: int = 3, rerank_top_n: int = 3) -> RetrieverEvent:
        st.info(f"Retrieving top-{top_k} documents for query: '{query}'")
        if not self.vector_index:
            st.error("Vector index is not loaded.")
            return RetrieverEvent(nodes=[], original_query=query)

        retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=top_k)
        nodes_with_scores = await retriever.aretrieve(query)

        if st.session_state.get("enable_reranker"):
            nodes_with_scores = functional_reranker(nodes_with_scores, query, rerank_top_n)

        return RetrieverEvent(nodes=nodes_with_scores, original_query=query)

    @step
    async def synthesize(self, ev: RetrieverEvent) -> StopEvent:
        if not ev.nodes:
            return StopEvent(result={"response": "No relevant info found.", "judgement": "N/A"})

        response_synthesizer = CompactAndRefine(llm=self.llm, streaming=False)
        try:
            response_obj = await response_synthesizer.asynthesize(ev.original_query, ev.nodes)
            response_text = response_obj.response
        except Exception as e:
            st.error(f"Synthesis failed: {e}")
            response_text = "Error generating response."

        judgement_text = "LLM-as-a-Judge disabled."
        if st.session_state.get("enable_llm_judge") and st.session_state.get("groq_api_key_sidebar"):
            judgement_text = await functional_llm_as_judge(
                query=ev.original_query,
                response_text=response_text,
                judge_llm_name=st.session_state.get("selected_model"),
                groq_api_key=st.session_state["groq_api_key_sidebar"]
            )

        return StopEvent(result={"response": response_text, "judgement": judgement_text})