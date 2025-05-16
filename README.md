# 🔬 Advanced RAG Comparison and Analysis

Welcome to the **Advanced RAG Comparison and Analysis Lab**! 

This Streamlit-based sandbox allows AI enthusiasts, students, and researchers to explore the internals of Retrieval Augmented Generation (RAG).


<i>Stay tuned! More advanced RAG techniques & features are underway and will be added soon.</i>



## 🚀 Highlights

* **Interactive Playground** to tweak and test RAG components.
* **Side-by-Side Comparison** of various indexing, retrieval, and LLM configurations.
* **Detailed Metadata Inspection** to understand each processing stage.

## ✨ Core Features

* **Indexing Backbones:** Vector Store & Knowledge Graph indexes
* **LLM Selection:** Choose models like Llama3, DeepSeek-r1-distill-llama-70b (via Groq API)
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
git clone https://github.com/MohanKrishnaGR/LLM.git
cd LLm
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r LLM/requirements.txt
```

### Run

```bash
streamlit run LLM/main.py
```

## How to Use

1. **Add Documents:** Place files in `LLM/data_for_rag/`.
2. **Configure Sidebar:** Choose LLM, index type, chunk size, etc.
3. **Build Index:** Click 'Create / Update Index'.
4. **Experiment:** Adjust retrieval & transformation settings.
5. **Query & Analyze:** Run queries and view detailed metadata.


## 👤 About the Developer

Built by **Mohan Krishna G R**, passionate about GenAI and RAG systems.

* [LinkedIn](https://www.linkedin.com/in/grmk/)
* [Portfolio](https://mohankrishnagr.github.io/)

---