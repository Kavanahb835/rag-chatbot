
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Document Chatbot")
st.write("Ask questions based on your uploaded documents")

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )

db = load_db()
llm = load_llm()

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Searching documents..."):
        docs = db.similarity_search(question, k=3)
        context = "\n".join([d.page_content for d in docs])
        prompt = f"Answer based on context only.\nContext: {context}\nQuestion: {question}\nAnswer:"
        result = llm(prompt)[0]['generated_text']

        st.subheader("Answer:")
        st.write(result)

        st.subheader("Source chunks used:")
        for i, doc in enumerate(docs):
            with st.expander(f"Chunk {i+1}"):
                st.write(doc.page_content)
