import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


import os


os.environ['GROQ_API_KEY']=st.secrets["GROQ_API_KEY"]

# Defining the UI

st.title("QnA Chatbot over PDF using llama")
llm=ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"],model_name="llama-3.3-70b-versatile")



#Prompt Engineering

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question strictly according to the context.
    Provide the most precise and accurate answer.
    <context>
    {context}
    </context>
    Question: {question}
    """
)

from langchain_community.document_loaders import PyPDFLoader
uploaded_file = st.file_uploader("📂 Upload a PDF file", type="pdf")

def vector_embedding():
  if"vectors" not in st.session_state:
    if uploaded_file is None:
      st.error("❌ Please upload a PDF first!")
      st.stop()
    with open("temp.pdf", "wb") as f:
      f.write(uploaded_file.read())
    st.session_state.embeddings=HuggingFaceEmbeddings(
    model_name="hkunlp/instructor-large")
    st.session_state.loader=PyPDFLoader("temp.pdf")#DataIngestion
    st.session_state.documents=st.session_state.loader.load()#Document Loading
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)#Chunk Creation
    st.session_state.texts=st.session_state.text_splitter.split_documents(st.session_state.documents)#Splitting
    st.session_state.vectorstore=FAISS.from_documents(st.session_state.texts,st.session_state.embeddings)#vector OpenAI embeddings
    st.session_state.vectors = True



prompt1=st.text_input("Enter your question")

if st.button("Document Embedding"):
  vector_embedding()
  st.write("Embedding Done")

import time

if prompt1:
   chain=create_stuff_documents_chain(llm,prompt)
   retriever=st.session_state.vectorstore.as_retriever()
   retrieval_chain=create_retrieval_chain(retriever,chain)
   start=time.process_time()
   try:
    response=retrieval_chain.invoke({'question':prompt1})
    st.write(f"Response time: {time.process_time()-start:.2f} seconds")
   except Exception as e:
    st.error(f"⚠️ Error while querying Groq LLM: {e}")
    st.stop()

   st.write(response['answer'])
 
