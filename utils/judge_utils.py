import streamlit as st
from .llm_utils import get_llm
from config import LLAMA4_MODEL

async def functional_llm_as_judge(query: str, response_text: str, judge_llm_name: str, groq_api_key: str):
    st.info(f"Evaluating with {judge_llm_name}")
    judge_llm = get_llm(judge_llm_name, groq_api_key)
    if not judge_llm:
        return "Judge LLM unavailable."

    prompt = f"""You are evaluating this response...

User Query:
"{query}"

Generated Response:
"{response_text[:2000]}"

Evaluate on relevance, coherence, helpfulness, conciseness, factuality.
Return:
Overall Score (1-5): ...
Evaluation Summary: ...
Detailed Justification: ..."""

    try:
        judgement = await judge_llm.acomplete(prompt)
        return judgement.text
    except Exception as e:
        st.warning(f"Judge evaluation failed: {e}")
        return str(e)
