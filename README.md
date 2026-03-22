# RAG Document Q&A Chatbot

A RAG (Retrieval-Augmented Generation) based document chatbot that answers questions from your own PDF documents using LangChain, FAISS, and Streamlit.

## Live Demo
[Click here to try the chatbot](#) <!-- Add your Streamlit Cloud link here after deployment -->

## What it does
Upload any PDF document and ask questions about it. The chatbot finds the most relevant sections from your document and generates accurate answers using a free LLM.

## Tech Stack
- **Python** — Core language
- **LangChain** — Document loading and pipeline
- **FAISS** — Vector database for semantic search
- **HuggingFace** — Free embedding model (all-MiniLM-L6-v2) and LLM (flan-t5)
- **Streamlit** — Web interface
- **PyPDF** — PDF text extraction

## How it works
1. PDF is loaded and split into 500 character chunks
2. Each chunk is converted to a vector using sentence-transformers
3. Vectors are stored in FAISS database
4. User question is converted to a vector
5. FAISS finds top 3 most similar chunks
6. Chunks + question sent to LLM
7. LLM generates a grounded answer

## Project Structure
```
rag-chatbot/
├── app.py              # Streamlit UI
├── ingest.py           # PDF loading and FAISS index creation
├── requirements.txt    # All dependencies
└── README.md           # Project documentation
```

## How to Run Locally
```bash
pip install -r requirements.txt
python ingest.py
streamlit run app.py
```

## Sample Questions Tested
- What tools and libraries are used for cloth try-on?
- What is GMM and TOM in virtual trial room?
- What is the project vision and mission?

## Developer
**Kavana H B**
MCA Graduate | AWS Certified | Software Developer
Bangalore University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/kavana-hb-680525345)
