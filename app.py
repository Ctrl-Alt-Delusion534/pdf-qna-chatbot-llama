import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
import os
import time

#UI
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="💬", layout="wide")
st.title("💬 Your Friendly PDF Q&A Assistant")

st.markdown("Upload a PDF in the sidebar, and I’ll help you explore it with clear, concise answers.")

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ['GROQ_API_KEY'] = groq_api_key
except KeyError:
    st.error(" Oops! I couldn’t find a `GROQ_API_KEY` in your secrets. Please add it before continuing.")
    st.stop()

# Prompt Template
prompt_template = """
Answer the question strictly from the context.
If you don’t find the answer, say "I don't have information about that in this document."

<context>
{context}
</context>

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

#Using HuggingFaceEmbeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="hkunlp/instructor-large")

# --- Session state setup ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("📂 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.vector_store = None
            st.session_state.retrieval_chain = None
            st.session_state.messages = []

            with st.spinner(" Reading and preparing your document... hang tight!"):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                embeddings = load_embeddings()
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

                llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
                doc_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector_store.as_retriever()
                st.session_state.retrieval_chain = create_retrieval_chain(retriever, doc_chain)

                st.success(" Done! Your PDF is ready — ask me anything from it.")

# --- Chat interface ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_q := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    if st.session_state.retrieval_chain is None:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please upload and process a PDF first so I can help.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                start = time.process_time()
                response = st.session_state.retrieval_chain.invoke({"input": user_q})
                elapsed = time.process_time() - start

                answer = response.get("answer", "Sorry, I couldn’t find an answer in the document.")
                # Add a friendlier tone
                final_answer = f"Here’s what I found:\n\n{answer}"

                st.markdown(final_answer)
                st.caption(f"⏱️ Response time: {elapsed:.2f} seconds")

                st.session_state.messages.append({"role": "assistant", "content": final_answer})
