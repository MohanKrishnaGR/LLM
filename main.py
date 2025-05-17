# LLM/main.py
# Main Streamlit application script

import streamlit as st
from utils.directory_utils import ensure_data_directory
from ui.sidebar import render_sidebar
from ui.main_ui import handle_main_ui # Corrected import: was run_async_main_ui

# Set page configuration for the Streamlit app
# This should be the first Streamlit command in your script, after imports.
st.set_page_config(
    page_title="Advanced RAG Comparison",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬" # Optional: Add a page icon
)

# Display the main title of the application
st.title("🔬 Advanced RAG Comparison and Analysis")

# Ensure the necessary data directory exists
ensure_data_directory()

# Render the sidebar for user configurations
render_sidebar()

# Handle the main UI elements and logic
handle_main_ui() # Corrected function call: was run_async_main_ui()

# Optional: Add a footer or other static content
st.markdown("---")
st.markdown(
    "Developed by **Mohan Krishna G R** | Exploring AI/RAG? Let's "
    "[connect on LinkedIn](https://www.linkedin.com/in/grmk/) | "
    "[Portfolio](https://mohankrishnagr.github.io/) "
)
st.markdown(
 # 🔬 Advanced RAG Comparison and Analysis Lab

Welcome to the **Advanced RAG Comparison and Analysis Lab**! This Streamlit-based sandbox allows AI enthusiasts, students, and researchers to explore the internals of Retrieval Augmented Generation (RAG).

Developed by [Mohan Krishna G R](https://mohankrishnagr.github.io/).

---

## 🚀 Highlights

* **Interactive Playground** to tweak and test RAG components.
* **Side-by-Side Comparison** of various indexing, retrieval, and LLM configurations.
* **Detailed Metadata Inspection** to understand each processing stage.

## ✨ Core Features

* **Indexing Backbones:** Vector Store & Knowledge Graph indexes
* **LLM Selection:** Choose models like Llama3, DeepSeek Coder (via Groq API)
* **Retrieval Techniques:** Vector-only, Sparse (BM25), and Hybrid search
* **Query Optimization:** Enable Query Transformations like HyDE
* **Re-ranking:** Use sentence transformer models to refine results
* **LLM-as-a-Judge:** Evaluate output coherence and relevance

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Framework:** LlamaIndex
* **LLMs:** Groq API (Llama3, DeepSeek Coder)
* **Embeddings:** FastEmbed
* **Rerankers:** Sentence Transformers
* **Sparse Retrieval:** BM25
* **Document Parsing:** PyMuPDF, python-docx, openpyxl, Markdown

## 📁 Folder Structure

```
LLM/
├── data_for_rag/         # Upload documents here
├── persisted_index/      # Stores indexes and document states
├── ui/                   # Streamlit UI components
├── utils/                # Helper functions
├── workflow/             # RAG logic (rag_workflow.py)
├── config.py             # App configuration
├── main.py               # Entry-point script
└── requirements.txt      # Dependencies
```

## 🚀 Setup Instructions

### Prerequisites

* Python 3.9+
* Groq API Key

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r LLM/requirements.txt
```

### Run

```bash
streamlit run LLM/main.py
```

## 🧪 How to Use

1. **Add Documents:** Place files in `LLM/data_for_rag/`.
2. **Configure Sidebar:** Choose LLM, index type, chunk size, etc.
3. **Build Index:** Click 'Create / Update Index'.
4. **Experiment:** Adjust retrieval & transformation settings.
5. **Query & Analyze:** Run queries and view detailed metadata.

## 🌱 Roadmap

* More reranker and LLM support
* Real-time visual metrics

## 👤 About the Developer

Built by **Mohan Krishna G R**, passionate about GenAI and RAG systems.

* [LinkedIn](https://www.linkedin.com/in/grmk/)
* [Portfolio](https://mohankrishnagr.github.io/)

## 📄 License

Add your preferred license (MIT, Apache 2.0, etc.)
   "<i>Stay tuned! More advanced RAG techniques & features are underway and will be added soon.</i>",
    unsafe_allow_html=True 
)

